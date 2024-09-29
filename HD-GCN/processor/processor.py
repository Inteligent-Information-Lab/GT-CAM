#!/usr/bin/env python
from __future__ import print_function

import torch.multiprocessing as mp
import argparse
import inspect
import os
import pickle
import random
import shutil
import sys
import time
from collections import OrderedDict
import traceback
from sklearn.metrics import confusion_matrix
import csv
import numpy as np
import glob
# import torchsnooper 
# torch
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm

from torchlight import DictAction

import matplotlib as mpl
mpl.use('TkAgg')

mp.set_sharing_strategy('file_system')

def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
   
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#@torchsnooper.snoop()
def import_class(import_str):
    #print(import_str)#import_str = 'model.HDGCN.Model'
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')
    
#@torchsnooper.snoop()
class Processor():
    """ 
        Network
    """
    #@torchsnooper.snoop()
    def __init__(self, argv=None):
        self.dev = "cuda:0"
        self.load_arg(argv)
        #print(self.load_arg(argv)) #None
        self.load_model()     
        self.load_data()
        self.load_optimizer()
        self.save_arg()
        if self.arg.phase == 'train':
            if not self.arg.train_feeder_args['debug']:
                self.arg.model_saved_name = os.path.join(self.arg.work_dir, 'runs')
                if os.path.isdir(self.arg.model_saved_name):
                    print('log_dir: ', self.arg.model_saved_name, 'already exist')
                    answer = input('delete it? y/n:')
                    if answer == 'y':
                        shutil.rmtree(self.arg.model_saved_name)
                        print('Dir removed: ', self.arg.model_saved_name)
                        input('Refresh the website of tensorboard by pressing any keys')
                    else:
                        print('Dir not removed: ', self.arg.model_saved_name)
                self.train_writer = SummaryWriter(os.path.join(self.arg.model_saved_name, 'train'), 'train')
                self.val_writer = SummaryWriter(os.path.join(self.arg.model_saved_name, 'val'), 'val')
            else:
                self.train_writer = self.val_writer = SummaryWriter(os.path.join(self.arg.model_saved_name, 'test'), 'test')
        self.global_step = 0
        # pdb.set_trace()
        #调用torch方法载入模型并加载模型权重参数
        self.load_model()

        self.lr = self.arg.base_lr
        self.best_acc = 0
        self.best_acc_epoch = 0

        #self.model = self.model.cuda(self.output_device)

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=self.output_device)
    
    #@torchsnooper.snoop()
    def load_arg(self, argv=None):
        
        parser = self.get_parser()
        #print(parser)
        # load arg form config file
        p = parser.parse_args(argv)
        #print(p)
        if p.config is not None:
            # load config file
            with open(p.config, 'r') as f:
                default_arg = yaml.load(f, Loader=yaml.FullLoader)#使用load方法加载文件

            # update parser from config file
            key = vars(p).keys()
            for k in default_arg.keys():
                if k not in key:
                    print('Unknown Arguments: {}'.format(k))
                    assert k in key

            parser.set_defaults(**default_arg)#parser.set_defaults函数将默认参数设置为加载的参数
        #print(argv)
        self.arg = parser.parse_args(argv)

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed)
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=True,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed)
         
 
    
    #@torchsnooper.snoop()
    def load_model(self):
        output_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        #output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device
        #self.model = self.procesoor.load_model(self.arg.model, **(self.arg.model_args))
        #print(self.model)
        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)

        self.model = Model(**self.arg.model_args)
        # print(self.model)
        
        if self.arg.loss_type == 'CE':
            self.loss = nn.CrossEntropyLoss().cuda(output_device)
            # self.loss = LabelSmoothingCrossEntropy(smoothing=0.1).cuda(output_device)
        else:
            #self.loss = LabelSmoothingCrossEntropy(smoothing=0.1).cuda(output_device)
            self.loss = nn.CrossEntropyLoss().cuda(output_device)
        
        if self.arg.weights:
           #print(arg.weights[:-3].split('-')[-1])
           #self.global_step = int(arg.weights[:-3].split('-')[-1])
           #self.global_step = float(arg.weights[:-3].split('-')[-1])
           self.print_log('Load weights from {}.'.format(self.arg.weights))
           if '.pkl' in self.arg.weights:
               with open(self.arg.weights, 'r') as f:
                   weights = pickle.load(f)
           else:
               weights = torch.load(self.arg.weights)

           weights = OrderedDict([[k.split('module.')[-1], v.cuda(output_device)] for k, v in weights.items()])

           keys = list(weights.keys())
           for w in self.arg.ignore_weights:
               for key in keys:
                   if w in key:
                       if weights.pop(key, None) is not None:
                           self.print_log('Sucessfully Remove Weights: {}.'.format(key))
                       else:
                           self.print_log('Can Not Remove Weights: {}.'.format(key))

           try:
               self.model.load_state_dict(weights)
           except:
               state = self.model.state_dict()
               diff = list(set(state.keys()).difference(set(weights.keys())))
               print('Can not find these weights:')
               for d in diff:
                   print('  ' + d)
               state.update(weights)
               self.model.load_state_dict(state)

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

        self.print_log('using warm up, epoch: {}'.format(self.arg.warm_up_epoch))

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            f.write(f"# command line: {' '.join(sys.argv)}\n\n")
            yaml.dump(arg_dict, f)


    def start(self):
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.print_log(f'# Parameters: {count_parameters(self.model)}')
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                save_model = (((epoch + 1) % self.arg.save_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch)) and (epoch+1) > self.arg.save_epoch

                self.train(epoch, save_model=True)

                # if epoch + 1 > 55:
                    
                self.eval(epoch, save_score=True, loader_name=['test'])

            # test the best model
            weights_path = glob.glob(os.path.join(self.arg.work_dir, 'runs-'+str(self.best_acc_epoch)+'*'))[0]
            weights = torch.load(weights_path)
            if type(self.arg.device) is list:
                if len(self.arg.device) > 1:
                    #weights = OrderedDict([['module.'+k, v.cuda(self.output_device)] for k, v in weights.items()])
                    weights = OrderedDict([['module.'+k, v] for k, v in weights.items()])
            self.model.load_state_dict(weights)

            wf = weights_path.replace('.pt', '_wrong.txt')
            rf = weights_path.replace('.pt', '_right.txt')
            self.arg.print_log = False
            self.eval(epoch=0, save_score=True, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.arg.print_log = True


            num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.print_log(f'Best accuracy: {self.best_acc}')
            self.print_log(f'Epoch number: {self.best_acc_epoch}')
            self.print_log(f'Model name: {self.arg.work_dir}')
            self.print_log(f'Model total number of params: {num_params}')
            self.print_log(f'Weight decay: {self.arg.weight_decay}')
            self.print_log(f'Base LR: {self.arg.base_lr}')
            self.print_log(f'Batch Size: {self.arg.batch_size}')
            self.print_log(f'Test Batch Size: {self.arg.test_batch_size}')
            self.print_log(f'seed: {self.arg.seed}')

        elif self.arg.phase == 'test':
            wf = self.arg.weights.replace('.pt', '_wrong.txt')
            rf = self.arg.weights.replace('.pt', '_right.txt')

            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            self.eval(epoch=0, save_score=self.arg.save_score, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.print_log('Done.\n')
    
    @staticmethod
    def get_parser(add_help=False):
    # parameter priority: command line > config > default
        parser = argparse.ArgumentParser(add_help=add_help, description='Network')
        parser.add_argument('--work-dir',default='./work_dir/temp',help='the work folder for storing results')

        parser.add_argument('-model_saved_name', default='')
        #test
        #parser.add_argument( '--config',default='./config/nturgbd-cross-view/bone_com_1test.yaml', help='path to the configuration file')
        #ntu60-xsub:/home/n303/YRR/HDGCN/config/nturgbd-cross-subject/bone_com_1test.yaml
        #ntu60-xview:/home/n303/YRR/HDGCN/config/nturgbd-cross-view/bone_com_1test.yaml
        #train
        #parser.add_argument( '--config',default='/home/jack/YRR/HDGCN/config/nturgbd-cross-view/joint_com_1.yaml', help='path to the configuration file')
        # parser.add_argument( '--config',default='./config/nturgbd-cross-subject/joint_com_1.yaml', help='path to the configuration file')
        #ntu60-xsub train:/home/jack/YRR/HDGCN/config/nturgbd-cross-subject/joint_com_1.yaml
        # train ntu120-setup:/home/jack/YRR/HDGCN/config/nturgbd120-cross-setup/bone_com_1.yaml
        #parser.add_argument( '--config',default='/data/home/st/GT_CAM/st-gcn/HD-GCN-main/config/ntu-xsub/bone_com_1_test.yaml', help='path to the configuration file')
        # parser.add_argument('--config', default='/data/home/st/GT_CAM/st-gcn/HD-GCN-main/config/ntu-xview/bone_com_1_test.yaml', help='path to the configuration file')
        parser.add_argument('--config', default='/data/home/st/GT_CAM/st-gcn/HD-GCN-main/config/ntu-csub/bone_com_1_test.yaml', help='path to the configuration file')
        # parser.add_argument('--config', default='/data/home/st/GT_CAM/st-gcn/HD-GCN-main/config/ntu-csetup/bone_com_1_test.yaml', help='path to the configuration file')

        # processor
        parser.add_argument( '--phase', default='test', help='must be train or test')
        parser.add_argument('--save-score',type=str2bool, default=False, help='if ture, the classification score will be stored')

        # visulize and debug
        parser.add_argument('--seed', type=int, default=1, help='random seed for pytorch')
        parser.add_argument( '--log-interval',type=int, default=100, help='the interval for printing messages (#iteration)')
        parser.add_argument('--save-interval',type=int,default=1, help='the interval for storing models (#iteration)')
        parser.add_argument('--save-epoch', type=int,default=30,help='the start epoch to save model (#iteration)')
        parser.add_argument( '--eval-interval',type=int,default=5,help='the interval for evaluating models (#iteration)')
        parser.add_argument( '--print-log', type=str2bool,default=True, help='print logging or not')
        parser.add_argument( '--show-topk', type=int, default=[1, 5], nargs='+', help='which Top K accuracy will be shown')

        # feeder
        parser.add_argument( '--feeder', default='feeder.feeder', help='data loader will be used')
        parser.add_argument( '--num-worker',type=int, default=4,  help='the number of worker for data loader')
        parser.add_argument( '--train-feeder-args', action=DictAction,  default=dict(), help='the arguments of data loader for training')
        parser.add_argument(  '--test-feeder-args',  action=DictAction,  default=dict(), help='the arguments of data loader for test')

        # model
        parser.add_argument('--model', default=None, help='the model will be used')
        parser.add_argument( '--model-args',action=DictAction,default=dict(),help='the arguments of model')
        #test
        # parser.add_argument('--weights',default='./data/ntu/ntu/cross-subject/bone_CoM_1/runs.pt',
        #     help='the weights for network initialization')
        #train
        parser.add_argument('--weights',default=None,
            help='the weights for network initialization')

        parser.add_argument( '--ignore-weights', type=str, default=[], nargs='+',  help='the name of weights which will be ignored in the initialization')


        # optim
        # parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        # parser.add_argument('--step', type=int, default=[], nargs='+',
        #                     help='the epoch where optimizer reduce the learning rate')
        # parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        # parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        # parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        # optim
        #parser.add_argument( '--base-lr', type=float, default=0.01, help='initial learning rate')
        # parser.add_argument( '--step',type=int,default=[20, 40, 60],nargs='+',help='the epoch where optimizer reduce the learning rate')
        # parser.add_argument('--device',type=int,default=0,nargs='+',help='the indexes of GPUs for training or testing')
        # parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        # parser.add_argument('--nesterov', type=str2bool, default=False, help='use nesterov or not')
        # parser.add_argument('--batch-size', type=int, default=256, help='training batch size')
        # parser.add_argument('--test-batch-size', type=int, default=256, help='test batch size')
        # parser.add_argument('--start-epoch',type=int,default=0,help='start training from which epoch')
        # parser.add_argument( '--num-epoch',type=int,default=80,help='stop training in which epoch')
        # parser.add_argument('--weight-decay', type=float,default=0.0005,help='weight decay for optimizer')
        # parser.add_argument(  '--lr-ratio', type=float,default=0.001, help='decay rate for learning rate')
        # parser.add_argument('--lr-decay-rate',type=float, default=0.1,help='decay rate for learning rate')
        # parser.add_argument('--warm_up_epoch', type=int, default=0)
        # parser.add_argument('--loss-type', type=str, default='CE')
        return parser


