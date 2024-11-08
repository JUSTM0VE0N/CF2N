# -*- encoding: utf-8 -*-
'''
@File    :   config.py
@Time    :   2023/03/29 12:13:41
@Author  :   Liuch(laxio) 
@Version :   1.0
@Source  :   own
@Note    :   get the configuration file and convert to *.yaml file
@Contact :   isliuch@yeah.net
'''

# here put the import lib
import re, yaml, os  

def get_config(cfg_path):
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
       u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.') 
    )
    with open(cfg_path, 'r') as f:
        cfg = yaml.load(f, Loader=loader)
    return cfg

def save_yml(info, cfg_path):
    with open(cfg_path, 'w') as f:
        yaml.dump(info, f, Dumper=yaml.SafeDumper)

if __name__ == '__main__':
    config = get_config('./option.yml')
    #print(config)
