#!/usr/bin/env python
import argparse
import sys

# torchlight
# import torchlight
from torchlight import import_class
from torchlight import DictAction
# from torchlight.torchlight.util import import_class
# from torchlight.torchlight.util import DictAction
import torch
import numpy as np
import random
import yaml

import matplotlib

matplotlib.use('Agg')


def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Processor collection')
    processors = dict()
    processors['recognition'] = import_class('processor.recognition.REC_Processor')
    # processors['ExplainerHdgcn'] = import_class('processor.ExplainerHdgcn.Explainer')
    processors['ShapleyCam'] = import_class('processor.ShapleyCam.Explainer')
    
    # add sub-parser
    subparsers = parser.add_subparsers(dest='processor')
    # print(subparsers)
    for k, p in processors.items():
        subparsers.add_parser(k, parents=[p.get_parser()])
        # print(k)
        # print(p)
    arg = parser.parse_args()
    # Processor = processors[arg.processor]
    Processor = processors['ShapleyCam']
    # Processor = processors['ExplainerHdgcn']

    # # start
    # Processor = processors['recognition']
    # #Processor = processors[arg.processor]

    # #print(sys.argv[2:])#取argv中的第2+1项值 为[]

    p = Processor(sys.argv[2:])

    p.start()

