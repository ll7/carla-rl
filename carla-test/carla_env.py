from gym import Env
from gym.spaces import Box
import numpy as np
import carla
import time
import random

import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class CarlaWalkerEnv(Env):
    """simple walk gym environment for carla where you try to walk as far as possible"""

    def __init__(
        self,
        render=True,
        verbose=True
    ):
        """establish connection to carla server and choose map"""

        # set constant parameters
        self.to_render = render
        self.verbose = verbose
        self.town = 'Town01'
        self.image_size_x = 128
        self.image_size_y = 128
        self.pov = 170.0

        self.actor_list = []

        # TODO this needs to be normalized later
        self.action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # receive a segmentation image from the carla server
        # https://carla.readthedocs.io/en/latest/ref_sensors/#semantic-segmentation-camera
        # TODO ensure only the Red value is returned as observation
        self.observation_space = Box(
            low=0,
            high=22,  # 22 is the number of classes in the segmentation image
            shape=(self.image_size_x, self.image_size_y, ),
            dtype=np.uint8
        )

        # carla setup
        
        logging.debug('waiting for server')
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(30.0)
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
        self.settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(self.settings)

        # set synchronous mode
        # https://carla.readthedocs.io/en/latest/adv_synchrony_timestep/#client-server-synchrony
        self.settings = self.world.get_settings()
        self.settings.synchronous_mode = True  # Enables synchronous mode
        self.world.apply_settings(self.settings)
        # TODO remember to make each step tick

        self.blueprint_library = self.world.get_blueprint_library()

        pass

    def __create_camera(self):
        """create a camera and attach it to the walker"""

        self.seg_camera_bp = self.blueprint_library.find(
            'sensor.camera.semantic_segmentation')

        self.camera_transform = carla.Transform(
            carla.Location(z=2.0),
            carla.Rotation(pitch=-90.0)
        )

        self.seg_camera_bp.set_attribute('image_size_x', '128')
        self.seg_camera_bp.set_attribute('image_size_y', '128')
        self.seg_camera_bp.set_attribute('fov', '170.0')

        self.camera = self.world.spawn_actor(
            self.seg_camera_bp,
            self.camera_transform,
            attach_to=self.walker
        )

        self.actor_list.append(self.camera)
        logging.debug('created %s' % self.camera.type_id)

    def __get_camera_image(self):
        """get the camera image and pass the camera image to the observation space"""

        pass

    def reset(self):
        """spawn walker and attach camera"""
        self.world.tick()
        self.actor_list = []

        self.walker_bp = self.blueprint_library.filter('0012')[0]

        spawn_points = self.world.get_map().get_spawn_points()

        self.walker_spawn_transform = random.choice(spawn_points)
        logging.debug('spawning walker at %s' %
                      self.walker_spawn_transform.location)

        self.walker = self.world.spawn_actor(
            self.walker_bp, self.walker_spawn_transform)

        self.actor_list.append(self.walker)
        logging.debug('created %s' % self.walker.type_id)
        
        # self.world.tick()
        if self.verbose:
            distance_from_spawn_point = self.walker.get_location() - \
                self.walker_spawn_transform.location
            logging.debug('distance from spawn point: %s' %
                          distance_from_spawn_point.length())

        # create camera

        # self.__create_camera()

        # TODO destroy camera
        self.observation = self.__get_camera_image()

        # set spectator if render is true
        if self.to_render == True:
            logging.debug('render per default during init')
            self.render(init=True)

        # make a tick for the changes to take effect
        # time.sleep(0.5)
        # for i in range(3):
        #     self.world.tick()
        #     logging.debug('tick {} in reset'.format(i))
        #     time.sleep(1)
        self.world.tick()
        logging.debug('tick in reset')

        return self.observation

    def step(self, action):
        """apply action to walker and return reward, observation, done, info"""
        logging.debug('taking step')

        self.world.tick()
        time.sleep(0.2) # TODO remove this

        # TODO return
        # return self.reward, self.observation, self.done, self.info


    def render(self, init = False):
        """show an rgb image of the current observation"""
        # TODO resetting the spectator only works, if there is a tick after the settings are applied
        
        logging.debug('rendering')
        # set spectator as top down view

        self.spectator = self.world.get_spectator()

        if init:
            # during the reset proces and the initial step, the walker is not spawned yet
            # therefore we would get an incorrect walker location and we choose the walker spawn location instead
            self.spectator_transform = self.walker_spawn_transform
        else:
            # if we are during the normal run process and want to set the spectator such that we have a correct view
            # we use the walker location
            self.spectator_transform = self.walker.get_transform() # only if you already ticked
        
        self.spectator_transform.location.z += 10.0
        self.spectator_transform.rotation.pitch = -90.0

        self.spectator.set_transform(self.spectator_transform)

    def close(self):
        """if done, tidy up and destroy actors"""
        logging.debug('closing environment')
        
        # self.camera.destroy() 
        # TODO why does the camera needs to be destroyed? TODO
        
        # logging.debug('active actors: {}'.format(self.world.get_actors()))
        logging.debug('listed actors: {}'.format(self.actor_list))
        self.client.apply_batch([carla.command.DestroyActor(x)
                                for x in self.actor_list])

        # tick for changes to take effect
        self.world.tick()

        logging.debug('destroyed actors')

        # self.settings.synchronous_mode = False  # Disable synchronous mode
        # self.world.apply_settings(self.settings)

        logging.debug('======== closed environment ========')