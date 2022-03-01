#!/usr/bin/env python3

from carla_env import CarlaWalkerEnv
import logging
import time


def main():
    env = CarlaWalkerEnv()
    env.reset()
    logging.debug('we start to take ticks in main')
    for i in range(1):
        env.step(env.action_space.sample())
        logging.debug('step {} in main'.format(i))
        # env.render()
        time.sleep(1)
    env.close()
    
    pass

if __name__=="__main__":
    main()