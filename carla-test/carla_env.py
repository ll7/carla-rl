from gym import Env
from gym.spaces import Box
import numpy as np
import carla
import time

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
            time.sleep(5)
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
        
        
        pass
    
    def reset(self):
        """spawn walker and attach camera"""
        
        
        pass
    
    def step(self, action):
        """apply action to walker and return reward, observation, done, info"""
        pass
        
    def render(self):
        """show an rgb image of the current observation"""
        pass
    
    def close(self):
        """if done, tidy up and destroy actors"""
        pass
    
