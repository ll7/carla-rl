from gym import Env
from gym.spaces import Box, Dict
import numpy as np
import carla
import time
from datetime import datetime
import random

import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def read_IP_from_file(file_name="ip-host.txt"):
    """read IP from file"""
    with open(file_name) as f:
        lines = f.readlines()
    IP = lines[0]
    print('IP: {}'.format(IP))
    return IP


class CarlaWalkerEnv(Env):
    """
    simple walk gym environment for carla where you try to walk as far as possible

    https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html
    """

    def __init__(
        self,
        render: bool = True,
        verbose: bool = True,
        host: str = 'localhost',
        town: str = 'Town01',
    ):
        """
        initializes the CarlaWalkerEnv Gym Environment

        Args:
            render (bool, optional): should the spectator view move with the vehicle. Defaults to True.
            verbose (bool, optional): Log everything. Defaults to True.
            host (str, optional): IP address of the carla server. Defaults to 'localhost'.
            town (str, optional): town to load. Defaults to 'Town01'.

        init:
        - define default values
        - define interface 
        - establish connection to carla server
        - choose map
        - spawn walker
        - attach camera
        """

        super(CarlaWalkerEnv, self).__init__()

        # set constant parameters
        self.to_render = render
        self.verbose = verbose
        self.host = host

        self.slow_step = False
        """sleep each step to show simulation in real time
        
        bad for training but good for debugging"""

        self.town = town
        self.image_size_x = 32
        self.image_size_y = 32
        self.fov = 170.0
        """field of view of the camera"""

        self.fixed_time_step = 0.05
        """fixed time step size in seconds
        used in synchronous mode"""

        self.max_tick_count = 20 * (1/self.fixed_time_step)
        """max number of ticks per episode"""

        self.max_walking_speed = 15.0 / 3.6  # m/s
        """max walking speed of the walker"""

        self.now = datetime.now().strftime('%Y-%m-%d_%H%M')

        self.actor_list = []
        """stores all actors that are created"""

        # === Space Definiton ===

        self.action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        """this needs to be normalized later to a total length of 1"""

        self.max_dist_start = 100.0
        """maximum distance to walk in one episode for observation space
        TODO check max_dist_start in episode end"""

        self.observation_space = Dict({
            "segmentation_camera":
                Box(
                    low=0,
                    high=22,  # 22 is the number of classes in the segmentation image
                    shape=(self.image_size_x, self.image_size_y, ),
                    dtype=np.uint8
                ),
            "distance_from_start": Box(
                low=-self.max_dist_start,
                high=self.max_dist_start,
                shape=(2,),
                dtype=np.float32
                ),
            # TODO fix low and high and shape
            "current_orientation": Box(low=-1, high=1, shape=(2,), dtype=np.float32),
        })
        """
        observation_space: gym.spaces.Dict
        
        receive a segmentation image from the carla server
        https://carla.readthedocs.io/en/latest/ref_sensors/#semantic-segmentation-camera
        """

        # TODO fix observation space everywhere else.

        # initialize observation by sampling from the observation space
        self.observation = self.observation_space.sample()

        self.__setup_carla()

        # === walker ===
        self.__spawn_walker()

        if self.verbose:
            # print distance to spawn point
            # TODO obsolete?
            distance_from_spawn_point = self.walker.get_location() - \
                self.walker_spawn_transform.location
            logging.debug('distance from spawn point: %s' %
                          distance_from_spawn_point.length())

        # === camera ===
        self.__create_camera()

        # create segmentation camera listener which is called each tick and updates observation
        self.camera.listen(
            lambda data: self.__create_observation(data)
        )

        # tick once for the changes to take effect
        self.world.tick()

    def __setup_carla(self):
        """
        setup carla server:

        - connect the client to the carla server
        - load the map
        - set synchronous mode
        """
        logging.debug('waiting for server')
        logging.debug('try to connect to server %s' % self.host)
        self.client = carla.Client(self.host, 2000)
        self.client.set_timeout(60.0)
        logging.debug('server connected')

        self.world = self.client.get_world()

        # set correct map
        self.map = self.world.get_map()
        if not self.map.name.endswith(self.town):
            logging.info("We want to use {}, but the map is named {}".format(
                self.town, self.map.name))
            self.world = self.client.load_world('Town01')
            while not self.world.get_map().name.endswith(self.town):
                logging.debug('{} not loaded yet'.format(self.town))
                time.sleep(0.5)
            # time.sleep(20)
            self.world = self.client.get_world()
            self.map = self.world.get_map()
        logging.info("Map {} loaded".format(self.map.name))

        # set fixed time-step
        # https://carla.readthedocs.io/en/latest/adv_synchrony_timestep/#fixed-time-step
        self.settings = self.world.get_settings()
        self.settings.fixed_delta_seconds = self.fixed_time_step
        self.world.apply_settings(self.settings)

        # set synchronous mode
        # https://carla.readthedocs.io/en/latest/adv_synchrony_timestep/#client-server-synchrony
        self.settings = self.world.get_settings()
        self.settings.synchronous_mode = True  # Enables synchronous mode
        self.world.apply_settings(self.settings)
        # TODO remember to make each step tick

        self.blueprint_library = self.world.get_blueprint_library()

    def __create_camera(self):
        """create a camera and attach it to the walker"""

        self.seg_camera_bp = self.blueprint_library.find(
            'sensor.camera.semantic_segmentation')

        self.camera_transform = carla.Transform(
            carla.Location(z=2.0),
            carla.Rotation(pitch=-90.0)
        )

        self.seg_camera_bp.set_attribute(
            'image_size_x', str(self.image_size_x))
        self.seg_camera_bp.set_attribute(
            'image_size_y', str(self.image_size_y))
        self.seg_camera_bp.set_attribute('fov', str(self.fov))

        self.camera = self.world.spawn_actor(
            self.seg_camera_bp,
            self.camera_transform,
            attach_to=self.walker
        )

        self.actor_list.append(self.camera)
        logging.debug('created %s' % self.camera.type_id)

    def __get_camera_image(self):
        """get the camera image and pass the camera image to the observation space"""
        # TODO delete because obsolete
        self.camera.listen(lambda image: image.save_to_disk(
            './tmp/{}/{}.png'.format(self.now, image.frame_number),
            carla.ColorConverter.CityScapesPalette)
        )
        # return observation

        pass

    def __create_observation(self, data: carla.Image):
        """create observation based on the camera image received

        Args:
            data (carla.Image): Image from the data

        the data we receive is a carla.Image object formatted as a bgra byte array

        bgra1 bgra2 ...

        we need only the red values according to
        https://carla.readthedocs.io/en/latest/ref_sensors/#semantic-segmentation-camera

        to get our observation we access the array with [start:end:step]
        and we finally reshape the array from 1D to 2D + R channel
        """

        logging.debug('=== creating observation ===')

        if self.verbose:
            data.save_to_disk('./tmp/{}/{}.png'.format(self.now,
                              data.frame_number), carla.ColorConverter.CityScapesPalette)
            logging.debug('image saved to disk')
        only_red = np.array(data.raw_data, dtype=np.uint8)[
            2::4]  # only red values from bgRa
        
        # reshape. validation: https://github.com/ll7/carla-rl/issues/7#issuecomment-1055582311
        self.observation["segmentation_camera"] = only_red.reshape(
            self.image_size_x,
            self.image_size_y
        )
        
        # TODO TEST if yaw is correct angle for orientation
        self.observation["current_orientation"] = self.walker.get_transform().rotation.yaw
        
        # TODO TEST if transform is recieved correctly
        # TODO location is 3D, observation is 2D.
        vec_from_spawn2walker = self.walker.get_transform().location - self.walker_spawn_transform.location
        
        self.observation["distance_from_start"] = self.walker.get_location() - \
            self.walker_spawn_transform.location
        

        logging.debug('=== observation created ===')

    def __reward_calculation(self):
        """calculate the reward and apply reward
        """
        self.reward = self.walker_spawn_transform.location.distance(
            self.walker.get_location())

    def __spawn_walker(self):
        """Spawn the walker in a random vehicle spawn point

        TODO make it possible to spawn the walker in a specific spawn point
        TODO choose a random spawn poitn for a **walker**        
        """

        self.walker_bp = self.blueprint_library.filter('0012')[0]

        spawn_points = self.world.get_map().get_spawn_points()

        self.walker_spawn_transform = random.choice(spawn_points)
        logging.debug('spawning walker at %s' %
                      self.walker_spawn_transform.location)

        self.walker = self.world.spawn_actor(
            self.walker_bp, self.walker_spawn_transform)

        self.actor_list.append(self.walker)
        logging.debug('created %s' % self.walker.type_id)

    def __set_walker_location(self):
        spawn_points = self.world.get_map().get_spawn_points()

        self.walker_spawn_transform = random.choice(spawn_points)

        self.walker.set_transform(self.walker_spawn_transform)

    def reset(self):
        """spawn walker and attach camera"""
        # self.world.tick() # why would I tick here?

        # self.actor_list = []

        # TODO check if the correct image is received and stored in observation or if the image is delayed

        # TODO destroy camera
        # self.observation = self.__get_camera_image()

        self.__set_walker_location()

        # set spectator if render is true
        if self.to_render == True:
            logging.debug('render per default during init')
            self.render(init=True)

        self.tick_count = 0
        self.reward = 0
        self.done = False
        self.info = {}

        self.world.tick()  # this has to be the last line in reset
        logging.debug('tick in reset')

        return self.observation

    def step(self, action: np.ndarray):
        """
        apply action to walker and return observation, reward, done, info

        Parameters
        ----------
        action : np.ndarray
            action of the walker

        Returns
        -------
        observation, reward, done, info : list
            _description_

        Example
        -------
        action: array([-0.612514  ,  0.95657253], dtype=float32)
            shape: (2,)

        ## Background

        carlawalkercontrol
        applies carla.WalkerControl https://carla.readthedocs.io/en/latest/python_api/
        """

        logging.debug('taking step')

        # self.walker_last_location = self.walker.get_location()

        # length of the action vector could effect the maximum speed of the walker
        # which is not desired and this special case is catched here
        action_length = np.linalg.norm(action)
        if action_length == 0.0:
            # the chances are slim, but theoretically both actions could be 0.0
            unit_action = np.array([0.0, 0.0], dtype=np.float32)
        elif action_length > 1.0:
            # create a vector for the action with the length of zero
            unit_action = action / action_length
        else:
            unit_action = action

        direction = carla.Vector3D(
            x=float(unit_action[0]), y=float(unit_action[1]), z=0.0)

        walker_control = carla.WalkerControl(
            direction, speed=self.max_walking_speed)

        self.walker.apply_control(walker_control)

        #### TICK ####
        self.world.tick()
        self.tick_count += 1
        ##############

        self.__reward_calculation()

        if self.tick_count >= self.max_tick_count:
            self.done = True
            logging.debug('done')

        # slow down simulation in verbose mode
        # TODO desired?
        if self.slow_step == True:
            time.sleep(self.fixed_time_step)

        return self.observation, self.reward, self.done, self.info

    def render(self, init=False):
        """show an rgb image of the current observation"""
        # TODO resetting the spectator only works, if there is a tick after the settings are applied

        logging.debug('rendering')
        # set spectator as top down view

        self.spectator = self.world.get_spectator()

        if init:
            # during the reset proces and the initial step, the walker is not spawned yet
            # therefore we would get an incorrect walker location and we choose the walker
            # spawn location instead
            self.spectator_transform = self.walker_spawn_transform
        else:
            # if we are during the normal run process and want to set the spectator
            # such that we have a correct view
            # we use the walker location
            self.spectator_transform = self.walker.get_transform()  # only if you already ticked

        self.spectator_transform.location.z += 10.0
        self.spectator_transform.rotation.pitch = -90.0

        self.spectator.set_transform(self.spectator_transform)

    def close(self):
