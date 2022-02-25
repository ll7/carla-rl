#!/usr/bin/env python3

from carla_env import CarlaWalkerEnv
import logging
import time


def main():
    env = CarlaWalkerEnv()
    env.reset()
    env.render()
    # env.step()
    time.sleep(5)
    env.close()
    
    pass

if __name__=="__main__":
    main()