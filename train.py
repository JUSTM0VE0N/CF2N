# -*- encoding: utf-8 -*-
'''
@File    :   train.py
@Time    :   2023/04/03 02:18:34
@Author  :   Liuch(laxio) 
@Version :   1.0
@Source  :   own
@Note    :   train
@Contact :   isliuch@yeah.net
'''

# here put the import lib
from utils.config import get_config
from solver.solver import Solver
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fra4PS===>training')
    parser.add_argument('--option_path', type=str, default='option.yml')
    opt = parser.parse_args()
    cfg = get_config(opt.option_path)
    solver = Solver(cfg)
    solver.run()
    