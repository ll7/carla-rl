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
    
    def __init__(self):
        """establish connection to carla server and choose map"""
        
        # set constant parameters
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
            high=22, # 22 is the number of classes in the segmentation image
            shape=(self.image_size_x, self.image_size_y, ), 
            dtype=np.uint8
            )
        
        # carla setup
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(2.0)
        
        self.world = self.client.get_world()
        
        # set correct map
        self.map = self.world.get_map()
        if not self.map.name.endswith(self.town):
            logging.info("We want to use {}, but the map is named {}".format(self.town, self.map.name))
            self.world = self.client.load_world('Town01')
            time.sleep(10)
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
        self.settings.synchronous_mode = True # Enables synchronous mode
        self.world.apply_settings(self.settings)
        # TODO remember to make each step tick
        
        self.blueprint_library = self.world.get_blueprint_library()

        
        pass
    
    def __get_camera_image(self):
        """get the camera image and pass the camera image to the observation space"""
        
        
        pass
        
    
    def reset(self):
        """spawn walker and attach camera"""
        
        self.actor_list = []
        
        self.walker_bp = self.blueprint_library.filter('0012')[0]
        
        self.walker_spawn_transform = random.choice(self.world.get_map().get_spawn_points())
        
        self.walker = self.world.spawn_actor(self.walker_bp, self.walker_spawn_transform)
        
        self.actor_list.append(self.walker)
        logging.debug('created %s' % self.walker.type_id)
        
        
        # create camera
        self.seg_camera_bp = self.blueprint_library.find('sensor.camera.semantic_segmentation')
        
        self.camera_transform = carla.Transform(carla.Location(z=2.0), carla.Rotation(pitch=-90.0))
        
        self.seg_camera_bp.set_attribute('image_size_x', '128')
        self.seg_camera_bp.set_attribute('image_size_y', '128')
        self.seg_camera_bp.set_attribute('fov', '170.0')
        
        self.camera = self.world.spawn_actor(self.seg_camera_bp, self.camera_transform, attach_to=self.walker)
        
        
        
        self.actor_list.append(self.camera)
        logging.debug('created %s' % self.camera.type_id)
        
        # we need to tick to make the walker appear
        self.world.tick()
        
        # TODO destroy camera
        self.observation = self.__get_camera_image()
        
        return self.observation
        
        
        pass
    
    def step(self, action):
        """apply action to walker and return reward, observation, done, info"""
        
        # return self.reward, self.observation, self.done, self.info
        
        # self.world.step()
        pass
        
    def render(self):
        """show an rgb image of the current observation"""
        logging.debug('rendering')
        # set spectator as top down view
        
        self.spectator = self.world.get_spectator()
        
        self.spectator_transform = self.walker.get_transform()
        
        self.spectator_transform.location.z += 10.0
        self.spectator_transform.rotation.pitch = -90.0
        
        self.spectator.set_transform(self.spectator_transform)
        
    
    def close(self):
        """if done, tidy up and destroy actors"""
        self.camera.destroy()
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
        logging.debug('destroyed actors')
        

        pass
    
