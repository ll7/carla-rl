#!/usr/bin/env python3

from carla_env import CarlaWalkerEnv
import logging
import time
from datetime import datetime # required for logging to file

from stable_baselines3.common import env_checker

def manual_iteration(env, number_of_iterations=1):
    """iterate manually over the environment"""
    logging.debug('manual_iteration for {} iterations'.format(number_of_iterations))
    print(env.reset())
    logging.debug('we start to take ticks in main')
    for i in range(number_of_iterations):
        logging.debug('step {} in main'.format(i))
        observation, reward, done, info = env.step(env.action_space.sample())
        logging.debug('type of observation: {}'.format(type(observation)))
        
        # env.render()
        time.sleep(1)
    env.close()
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
    env = CarlaWalkerEnv(verbose=False)
    
    manual_iteration(env, number_of_iterations=3)
    
    logging.debug('stabel_baselines3 env_checker')
    env_checker.check_env(env)

    


if __name__ == "__main__":
    main()
