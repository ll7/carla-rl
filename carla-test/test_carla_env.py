#!/usr/bin/env python3

from carla_env import CarlaWalkerEnv
import logging
import time
from datetime import datetime


def main():

    # setup logging

    # logPath = './logs/logging/' + __name__
    # now = datetime.now().strftime('%Y-%m-%d_%H%M')
    # fileName = now + '_' + __name__ + '.log'

    # logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    # rootLogger = logging.getLogger()

    # fileHandler = logging.FileHandler("{0}/{1}.log".format(logPath, fileName))
    # fileHandler.setFormatter(logFormatter)
    # rootLogger.addHandler(fileHandler)

    # consoleHandler = logging.StreamHandler()
    # consoleHandler.setFormatter(logFormatter)
    # rootLogger.addHandler(consoleHandler)

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            # logging.FileHandler(
            #     "./tmp/logs/logging/" + datetime.now().strftime('%Y-%m-%d_%H%M') + ".log"),
            logging.StreamHandler()
        ]
    )

    env = CarlaWalkerEnv()
    print(env.reset())
    logging.debug('we start to take ticks in main')
    for i in range(10):
        env.step(env.action_space.sample())
        logging.debug('step {} in main'.format(i))
        # env.render()
        time.sleep(1)
    env.close()

    pass


if __name__ == "__main__":
    main()
