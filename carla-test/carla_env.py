from gym import Env
from gym.spaces import Box
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
    """simple walk gym environment for carla where you try to walk as far as possible

    https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html
    """

    def __init__(
        self,
        render=True,
        verbose=True,
        host='localhost',
    ):
        """establish connection to carla server and choose map"""

        # set constant parameters
        self.to_render = render
        self.verbose = verbose
        self.host = host

        self.slow_step = False  # sleep each step to show simulation in real time

        self.town = 'Town01'
        self.image_size_x = 32
        self.image_size_y = 32
        self.pov = 170.0

        self.max_tick_count = 20*20
        self.fixed_time_step = 0.05

        self.max_walking_speed = 15.0 / 3.6  # m/s

        self.observation = np.ndarray(
            shape=(self.image_size_x, self.image_size_y, ), dtype=np.uint8)

        self.now = datetime.now().strftime('%Y-%m-%d_%H%M')

        self.actor_list = []

        # this needs to be normalized later to a total length of 1
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
            # lambda image: image.save_to_disk(
            #     './tmp/{}/{}.png'.format(self.now, image.frame_number),
            #     carla.ColorConverter.CityScapesPalette
            #     )
        )

        # tick once for the changes to take effect
        self.world.tick()

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
        self.seg_camera_bp.set_attribute('fov', str(self.pov))

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

    def __create_observation(self, data):
        """create observation based on the camera image received

        the data we receive is a carla.Image object formatted as a bgra byte array

        bgra1 bgra2 ...

        we need only the red values according to
        https://carla.readthedocs.io/en/latest/ref_sensors/#semantic-segmentation-camera

        to get our observation we access the array with [start:end:step]
        and we finally reshape the array from 1D to 2D + R channel
        """
        logging.debug('=== creating observation ===')
        # logging.debug('data: {}'.format(data))
        # logging.debug('data type: {}'.format(type(data)))
        # logging.debug('data len: {}'.format(len(data)))
        # with open("data_raw_dat" + self.now + ".txt", "w") as text_file:
        #     text_file.write(str(data.raw_data))
        # data.convert(carla.ColorConverter.Raw)

        if self.verbose:
            data.save_to_disk('./tmp/{}/{}.png'.format(self.now,
                              data.frame_number), carla.ColorConverter.CityScapesPalette)
            logging.debug('image saved to disk')
        only_red = np.array(data.raw_data, dtype=np.uint8)[
            2::4]  # only red values from bgRa
        # reshape. validation: https://github.com/ll7/carla-rl/issues/7#issuecomment-1055582311
        self.observation = only_red.reshape(
            self.image_size_x, self.image_size_y)

        # self.observation = np.array(data.raw_data, dtype=np.uint8)[2::4] # only red values from bgRa
        # np.array(data.raw_data)
        # logging.debug('observation: {}'.format(self.observation))
        # logging.debug('observation type: {}'.format(type(self.observation)))
        # logging.debug('observation len: {}'.format(len(self.observation)))
        # with open("observation" + self.now + ".txt", "w") as text_file:
        #     text_file.write(str(self.observation))

        # np.savetxt("observation" + self.now + ".txt",
        #            self.observation, fmt='%d')

        # data.convert(carla.ColorConverter.Raw)
        # print(data)
        # print(len(data))
        # print(type(data))

        logging.debug('=== observation created ===')

    def __reward_calculation(self):
        """calculate the reward and apply reward
        """
        # distance_vector = self.walker_spawn_transform - self.walker.get_location()
        self.reward = self.walker_spawn_transform.location.distance(
            self.walker.get_location())

    def __spawn_walker(self):
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

    def step(self, action):
        """apply action to walker and return reward, observation, done, info

        action: array([-0.612514  ,  0.95657253], dtype=float32)
            shape: (2,)

        #carlawalkercontrol
        applies carla.WalkerControl https://carla.readthedocs.io/en/latest/python_api/
        """
        logging.debug('taking step')

        # self.walker_last_location = self.walker.get_location()

        # length of the action vector could effect the maximum speed of the walker which is not desired and this special case is catched here
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

        # TODO return
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


class CarlaVehicleEnv(Env):
    """simple drive gym environment for carla where you try to drive as far as possible

    https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html
    """

    def __init__(
        self,
        render=True,
        verbose=True,
        host='localhost',
    ):
        """establish connection to carla server and choose map"""

        # set constant parameters
        self.to_render = render
        self.verbose = verbose
        self.host = host

        self.slow_step = False  # sleep each step to show simulation in real time

        self.town = 'Town01'
        self.image_size_x = 32
        self.image_size_y = 32
        self.pov = 170.0

        self.max_tick_count = 20*20
        self.fixed_time_step = 0.05

        self.max_walking_speed = 15.0 / 3.6  # m/s

        self.observation = np.ndarray(
            shape=(self.image_size_x, self.image_size_y, ), dtype=np.uint8)

        self.now = datetime.now().strftime('%Y-%m-%d_%H%M')

        self.actor_list = []

        # this needs to be normalized later to a total length of 1
        self.action_space = Box(
            low=0.0, high=1.0, shape=(3,), dtype=np.float32)

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

        # === walker ===
        self.__spawn_vehicle()

        if self.verbose:
            # print distance to spawn point
            # TODO obsolete?
            distance_from_spawn_point = self.vehicle.get_location() - \
                self.vehicle_spawn_transform.location
            logging.debug('distance from spawn point: %s' %
                          distance_from_spawn_point.length())

        # === camera ===
        self.__create_camera()

        # === collision detection ===
        self.__collision_sensor()

        # === lane invasion detection ===
        self.__lane_invasion_sensor()

        # create segmentation camera listener which is called each tick and updates observation
        self.camera.listen(
            lambda data: self.__create_observation(data)
            # lambda image: image.save_to_disk(
            #     './tmp/{}/{}.png'.format(self.now, image.frame_number),
            #     carla.ColorConverter.CityScapesPalette
            #     )
        )

        # tick once for the changes to take effect
        self.world.tick()

    def __lane_invasion_sensor(self):
        lane_invasion_bp = self.blueprint_library.find(
            'sensor.other.lane_invasion')
        self.lane_invasion_sensor = self.world.spawn_actor(
            lane_invasion_bp,
            carla.Transform(),
            attach_to=self.vehicle)
        self.lane_invasion_sensor.listen(
            lambda event: self.__lane_invasion_callback(event))

    def __lane_invasion_callback(self, event):
        logging.debug('lane invasion detected')
        logging.debug(event)
        self.lane_invasion = True

    def __collision_sensor(self):
        collision_bp = self.blueprint_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(
            collision_bp,
            carla.Transform(),
            attach_to=self.vehicle
        )

        self.actor_list.append(self.collision_sensor)
        self.collision_sensor.listen(
            lambda event: self.__collision_sensor_callback(event))

    def __collision_sensor_callback(self, event):
        logging.debug('collision event: %s' % event)
        self.collision_detected = True

    def __create_camera(self):
        """create a camera and attach it to the walker"""

        self.seg_camera_bp = self.blueprint_library.find(
            'sensor.camera.semantic_segmentation')

        self.camera_transform = carla.Transform(
            carla.Location(z=2.0),
            carla.Rotation(pitch=-70.0)
        )

        self.seg_camera_bp.set_attribute(
            'image_size_x', str(self.image_size_x))
        self.seg_camera_bp.set_attribute(
            'image_size_y', str(self.image_size_y))
        self.seg_camera_bp.set_attribute('fov', str(self.pov))

        self.camera = self.world.spawn_actor(
            self.seg_camera_bp,
            self.camera_transform,
            attach_to=self.vehicle
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

    def __create_observation(self, data):
        """create observation based on the camera image received

        the data we receive is a carla.Image object formatted as a bgra byte array

        bgra1 bgra2 ...

        we need only the red values according to
        https://carla.readthedocs.io/en/latest/ref_sensors/#semantic-segmentation-camera

        to get our observation we access the array with [start:end:step]
        and we finally reshape the array from 1D to 2D + R channel
        """
        logging.debug('=== creating observation ===')
        # logging.debug('data: {}'.format(data))
        # logging.debug('data type: {}'.format(type(data)))
        # logging.debug('data len: {}'.format(len(data)))
        # with open("data_raw_dat" + self.now + ".txt", "w") as text_file:
        #     text_file.write(str(data.raw_data))
        # data.convert(carla.ColorConverter.Raw)

        if self.verbose:
            data.save_to_disk('./tmp/{}/{}.png'.format(self.now,
                              data.frame_number), carla.ColorConverter.CityScapesPalette)
            logging.debug('image saved to disk')
        only_red = np.array(data.raw_data, dtype=np.uint8)[
            2::4]  # only red values from bgRa
        # reshape. validation: https://github.com/ll7/carla-rl/issues/7#issuecomment-1055582311
        self.observation = only_red.reshape(
            self.image_size_x, self.image_size_y)

        # self.observation = np.array(data.raw_data, dtype=np.uint8)[2::4] # only red values from bgRa
        # np.array(data.raw_data)
        # logging.debug('observation: {}'.format(self.observation))
        # logging.debug('observation type: {}'.format(type(self.observation)))
        # logging.debug('observation len: {}'.format(len(self.observation)))
        # with open("observation" + self.now + ".txt", "w") as text_file:
        #     text_file.write(str(self.observation))

        # np.savetxt("observation" + self.now + ".txt",
        #            self.observation, fmt='%d')

        # data.convert(carla.ColorConverter.Raw)
        # print(data)
        # print(len(data))
        # print(type(data))

        logging.debug('=== observation created ===')

    def __reward_calculation(self):
        """calculate the reward and apply reward
        """
        self.reward = 0.0  # reset reward
        # distance_vector = self.walker_spawn_transform - self.walker.get_location()
        self.distance_travelled = self.vehicle_spawn_transform.location.distance(
            self.vehicle.get_location())
        self.reward += self.distance_travelled * 10.0

        absolute_acceleration = self.vehicle.get_acceleration().length()
        self.reward += absolute_acceleration * 10.0

        absolute_velocity = self.vehicle.get_velocity().length()
        self.reward += absolute_velocity ** 2

        if self.action[0] > 0.0 and self.action[2] > self.brake_limit:
            """punish if both throttle and braking are non-zero"""
            self.reward -= 10.0

        # punish steering
        steering_change = abs(self.action[1] - self.last_action[1])
        self.reward -= steering_change * 100.0

        # punish braking
        braking = self.action[2]
        self.reward -= braking * 10.0

        # punish angular velocity
        angular_velocity = self.vehicle.get_angular_velocity().length()
        self.reward -= angular_velocity * 10.0

        # reward throttle
        self.reward += self.action[0] * 1000.0

        # punish collision
        if self.collision_detected:
            self.reward -= 1000.0

        # punish lane invasion
        if self.lane_invasion_detected:
            self.reward -= 1000.0

    def __spawn_vehicle(self):
        self.vehicle_bp = self.blueprint_library.filter('model3')[0]

        spawn_points = self.world.get_map().get_spawn_points()

        self.vehicle_spawn_transform = random.choice(spawn_points)
        logging.debug('spawning walker at %s' %
                      self.vehicle_spawn_transform.location)

        self.vehicle = self.world.spawn_actor(
            self.vehicle_bp, self.vehicle_spawn_transform)

        self.actor_list.append(self.vehicle)
        logging.debug('created %s' % self.vehicle.type_id)

    def __set_vehicle_location(self):
        spawn_points = self.world.get_map().get_spawn_points()

        self.vehicle_spawn_transform = random.choice(spawn_points)

        self.vehicle.set_transform(self.vehicle_spawn_transform)

    def reset(self):
        """spawn walker and attach camera"""
        # self.world.tick() # why would I tick here?

        # self.actor_list = []

        # TODO check if the correct image is received and stored in observation or if the image is delayed

        # TODO destroy camera
        # self.observation = self.__get_camera_image()

        self.__set_vehicle_location()

        # set spectator if render is true
        if self.to_render == True:
            logging.debug('render per default during init')
            self.render(init=True)

        self.last_action = self.action_space.sample()

        self.collision_detected = False
        self.lane_invasion_detected = False

        self.tick_count = 0
        self.reward = 0
        self.done = False
        self.info = {}

        self.world.tick()  # this has to be the last line in reset
        logging.debug('tick in reset')

        return self.observation

    def step(self, action):
        """apply action to walker and return reward, observation, done, info

        action: array([-0.612514  ,  0.95657253], dtype=float32)
            shape: (2,)

        #carlawalkercontrol
        applies carla.WalkerControl https://carla.readthedocs.io/en/latest/python_api/
        """

        logging.debug('taking step')
        logging.debug('action: {}'.format(action))

        self.action = action
        # self.walker_last_location = self.walker.get_location()

        # length of the action vector could effect the maximum speed of the walker which is not desired and this special case is catched here
        # action_length = np.linalg.norm(action)
        # if action_length == 0.0:
        #     # the chances are slim, but theoretically both actions could be 0.0
        #     unit_action = np.array([0.0, 0.0], dtype=np.float32)
        # elif action_length > 1.0:
        #     # create a vector for the action with the length of zero
        #     unit_action = action / action_length
        # else:
        #     unit_action = action

        # direction = carla.Vector3D(
        #     x=float(unit_action[0]), y=float(unit_action[1]), z=0.0)

        vehicle_control = carla.VehicleControl()
        vehicle_control.throttle = float(action[0])
        vehicle_control.steer = float(action[1]*2.0-1.0)
        brake = float(action[2])
        self.brake_limit = 0.3
        if brake < self.brake_limit:
            """learn not to brake"""
            brake = 0.0
        vehicle_control.brake = brake
        vehicle_control.manual_gear_shift = False

        # walker_control = carla.WalkerControl(
        #     direction, speed=self.max_walking_speed)

        self.vehicle.apply_control(vehicle_control)

        #### TICK ####
        self.world.tick()
        self.tick_count += 1
        ##############

        self.__reward_calculation()

        self.collsion_detected = False
        self.lane_invasion_detected = False

        self.last_action = action
        if self.tick_count >= self.max_tick_count:
            self.done = True
            logging.info('distance travelled: {}'.format(
                self.distance_travelled))
            logging.debug('done')

        # slow down simulation in verbose mode
        # TODO desired?
        if self.slow_step == True:
            time.sleep(self.fixed_time_step)

        # TODO return
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
            self.spectator_transform = self.vehicle_spawn_transform
        else:
            # if we are during the normal run process and want to set the spectator
            # such that we have a correct view
            # we use the walker location
            self.spectator_transform = self.vehicle.get_transform()  # only if you already ticked

        self.spectator_transform.location.z += 100.0
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
