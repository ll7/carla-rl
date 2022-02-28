#!/usr/bin/env python3

from carla_env import CarlaWalkerEnv
import logging
import time


def main():
    env = CarlaWalkerEnv()
    env.reset()
    # env.render()
    # env.step(env.action_space.sample())
    time.sleep(5)
    # env.step(env.action_space.sample())
    for i in range(1):
        env.step(env.action_space.sample())
        logging.debug('step {} in main'.format(i))
        # env.render()
        time.sleep(1)
    env.close()
    
    pass

if __name__=="__main__":
    main()