#!/usr/bin/env python3

from carla_env import CarlaWalkerEnv, read_IP_from_file
import logging
import time
from datetime import datetime # required for logging to file
from stable_baselines3 import PPO

model_location = 'carla-test/tmp/train/logs/2022-03-03_1145/best_model_600000'

def manual_iteration(env, episode_length=20*60, episodes=5):
    """iterate manually over the environment"""
    logging.debug('manual_iteration for {} iterations'.format(episode_length))
    # print(env.reset())
    
    logging.debug('load model from {}'.format(model_location))
    model = PPO.load(model_location)
    
    for e in range(episodes):
        logging.debug('episode {}'.format(e))
        observation = env.reset()
        env.render()
        for s in range(episode_length):
            logging.debug('step {} in episode'.format(s, e))
            action, _ = model.predict(observation)
            observation, reward, done, info = env.step(action)
            logging.debug('type of observation: {}'.format(type(observation)))
            
            
            time.sleep(0.05)
        env.close()
        logging.debug('manual_iteration done')
    
    # logging.debug('we start to take ticks in main')
    # for s in range(episode_length):
    #     logging.debug('step {} in main'.format(s))
    #     observation, reward, done, info = env.step(env.action_space.sample())
    #     logging.debug('type of observation: {}'.format(type(observation)))
        
    #     # env.render()
    #     time.sleep(1)
    # env.close()
    logging.debug('manual_iteration done')
    
    
def main():
    """main function"""

    # set up logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            # logging.FileHandler(
            #     "./tmp/logs/logging/" + datetime.now().strftime('%Y-%m-%d_%H%M') + ".log"),
            logging.StreamHandler()
        ]
    )
    
    logging.info('=== start main ===')

    # set up environment
    env = CarlaWalkerEnv(verbose=False, host=read_IP_from_file())
    
    
    manual_iteration(env, episode_length=40)
    
if __name__ == "__main__":
    main()