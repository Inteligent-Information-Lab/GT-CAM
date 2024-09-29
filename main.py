#!/usr/bin/env python
import argparse
import sys
import warnings
import torch
num_devices = torch.cuda.device_count()
for i in range(num_devices):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")

# 忽略特定警告
warnings.filterwarnings("ignore", category=UserWarning, message="Using a non-full backward hook when the forward contains multiple autograd Nodes")

# 你的其他代码..
# torchlight
# import torchlight
from torchlight import import_class

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Processor collection')

    # region register processor yapf: disable
    processors = dict()
    processors['recognition'] = import_class('processor.recognition.REC_Processor')
    processors['ShapleyCam'] = import_class('processor.ShapleyCam.Explainer')
    
    #endregion yapf: enable


    # add sub-parser
    subparsers = parser.add_subparsers(dest='processor')
    for k, p in processors.items():
        subparsers.add_parser(k, parents=[p.get_parser()])

    # read arguments
    arg = parser.parse_args()
    arg.processor = 'ShapleyCam'

    # start
    Processor = processors[arg.processor]
    p = Processor(sys.argv[2:])


    p.start()
