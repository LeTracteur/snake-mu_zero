import utils
import argparse
import sys
from environment_new import *


def main(stg):
    settings = utils.Obj(stg)
    env = SnakeEnv(settings.env)
    env.reset()
    env.render()


    return env

if __name__ == '__main__':
    setting_file = sys.argv[1]
    stg_obj = utils.load_stgs(setting_file)
    a = main(stg_obj)
