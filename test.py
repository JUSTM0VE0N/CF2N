# -*- encoding: utf-8 -*-
'''
@File    :   test.py
@Time    :   2023/04/03 12:18:00
@Author  :   Liuch(laxio) 
@Version :   1.0
@Source  :   own
@Note    :   test
@Contact :   isliuch@yeah.net
'''

# here put the import lib
from utils.config  import get_config
from solver.testsolver import Testsolver
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fra4PS===>testing')
    parser.add_argument('--option_path', type=str, default='option.yml')
    opt = parser.parse_args()
    cfg = get_config(opt.option_path)
    solver = Testsolver(cfg)
    solver.run()
    