#!/usr/bin/env python
from __future__ import print_function

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
import torchsnooper 
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
from .processor import Processor
import matplotlib as mpl
mpl.use('TkAgg')
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

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
class REC_Processor(Processor):
    """ 
        Processor for Skeleton-based Action Recgnitions
    """      
    #@torchsnooper.snoop()
    def load_model(self):
        output_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device
        #print('!!!!!!!!!!!',self.arg.model,'!!!!!!!!!!!')
        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        #print('###########')
        #print(Model)
        self.model = Model(**self.arg.model_args)
        #print('self.model',self.model)
        if self.arg.loss_type == 'CE':
            #self.loss = nn.CrossEntropyLoss().cuda(output_device)
            self.loss = nn.CrossEntropyLoss()
            # self.loss = LabelSmoothingCrossEntropy(smoothing=0.1).cuda(output_device)
        else:
            #self.loss = LabelSmoothingCrossEntropy(smoothing=0.1).cuda(output_device)
            self.loss = nn.CrossEntropyLoss()
            #self.loss = nn.CrossEntropyLoss().cuda(output_device)
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

           #weights = OrderedDict([[k.split('module.')[-1], v.cuda(output_device)] for k, v in weights.items()])
           weights = OrderedDict([[k.split('module.')[-1], v] for k, v in weights.items()])

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

    def adjust_learning_rate(self, epoch, idx):
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                # lr = self.arg.base_lr * (
                #         self.arg.lr_decay_rate ** np.sum(epoch >= np.array(self.arg.step)))
                T_max = len(self.data_loader['train']) * (self.arg.num_epoch - self.arg.warm_up_epoch)
                T_cur = len(self.data_loader['train']) * (epoch - self.arg.warm_up_epoch) + idx

                eta_min = self.arg.base_lr * self.arg.lr_ratio
                lr = eta_min + 0.5 * (self.arg.base_lr - eta_min) * (1 + np.cos((T_cur / T_max) * np.pi))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def train(self, epoch, save_model=False):
        self.model.train()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        loader = self.data_loader['train']

        loss_value = []
        acc_value = []
        self.train_writer.add_scalar('epoch', epoch, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader)

        for batch_idx, (data, label, index) in enumerate(process):

            self.adjust_learning_rate(epoch, batch_idx)

            self.global_step += 1
            with torch.no_grad():
                # data = data.float().cuda(self.output_device)
                # label = label.long().cuda(self.output_device)
                data = data.float()
                label = label.long()
            timer['dataloader'] += self.split_time()

            # forward
            output = self.model(data)
            loss = self.loss(output, label)
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_value.append(loss.data.item())
            timer['model'] += self.split_time()

            value, predict_label = torch.max(output.data, 1)
            acc = torch.mean((predict_label == label.data).float())
            acc_value.append(acc.data.item())
            self.train_writer.add_scalar('acc', acc, self.global_step)
            self.train_writer.add_scalar('loss', loss.data.item(), self.global_step)

            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']
            self.train_writer.add_scalar('lr', self.lr, self.global_step)
            timer['statistics'] += self.split_time()

        # statistics of time consumption and loss
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log(
            '\tMean training loss: {:.4f}.  Mean training acc: {:.2f}%.'.format(np.mean(loss_value), np.mean(acc_value)*100))
        self.print_log('\tLearning Rate: {:.4f}'.format(self.lr))
        self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))

        if save_model:
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])

            torch.save(weights, self.arg.model_saved_name + '-' + str(epoch+1) + '-' + str(int(self.global_step)) + '.pt')

    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')
        self.model.eval()
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        for ln in loader_name:
            loss_value = []
            score_frag = []
            label_list = []
            pred_list = []
            step = 0
            process = tqdm(self.data_loader[ln])
            for batch_idx, (data, label, index) in enumerate(process):
                label_list.append(label)
                with torch.no_grad():
                    #data = data.float().cuda(self.output_device)
                    #label = label.long().cuda(self.output_device)
                    data = data.float()
                    label = label.long()
                    output = self.model(data)
                    loss = self.loss(output, label)
                    score_frag.append(output.data.cpu().numpy())
                    loss_value.append(loss.data.item())

                    _, predict_label = torch.max(output.data, 1)
                    pred_list.append(predict_label.data.cpu().numpy())
                    step += 1

                if wrong_file is not None or result_file is not None:
                    predict = list(predict_label.cpu().numpy())
                    true = list(label.data.cpu().numpy())
                    for i, x in enumerate(predict):
                        if result_file is not None:
                            f_r.write(str(x) + ',' + str(true[i]) + '\n')
                        if x != true[i] and wrong_file is not None:
                            f_w.write(str(index[i]) + ',' + str(x) + ',' + str(true[i]) + '\n')
            score = np.concatenate(score_frag)
            loss = np.mean(loss_value)
            if 'ucla' in self.arg.feeder:
                self.data_loader[ln].dataset.sample_name = np.arange(len(score))
            accuracy = self.data_loader[ln].dataset.top_k(score, 1)
            if accuracy > self.best_acc:
                self.best_acc = accuracy
                self.best_acc_epoch = epoch + 1

            print('Accuracy: ', accuracy, ' model: ', self.arg.model_saved_name)
            if self.arg.phase == 'train':
                self.val_writer.add_scalar('loss', loss, self.global_step)
                self.val_writer.add_scalar('acc', accuracy, self.global_step)

            score_dict = dict(
                zip(self.data_loader[ln].dataset.sample_name, score))
            self.print_log('\tMean {} loss of {} batches: {}.'.format(
                ln, len(self.data_loader[ln]), np.mean(loss_value)))
            for k in self.arg.show_topk:
                self.print_log('\tTop{}: {:.2f}%'.format(
                    k, 100 * self.data_loader[ln].dataset.top_k(score, k)))

            if save_score:
                with open('{}/epoch{}_{}_score.pkl'.format(
                        self.arg.work_dir, epoch + 1, ln), 'wb') as f:
                    pickle.dump(score_dict, f)

            # acc for each class:
            label_list = np.concatenate(label_list)
            pred_list = np.concatenate(pred_list)
            confusion = confusion_matrix(label_list, pred_list)
            list_diag = np.diag(confusion)
            list_raw_sum = np.sum(confusion, axis=1)
            each_acc = list_diag / list_raw_sum
            with open('{}/epoch{}_{}_each_class_acc.csv'.format(self.arg.work_dir, epoch + 1, ln), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(each_acc)
                writer.writerows(confusion)

    
    @staticmethod
    def get_parser(add_help=False):
    # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(add_help=add_help, parents=[parent_parser], description='Network')

        # optim
        parser.add_argument( '--base-lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument( '--step',type=int,default=[20, 40, 60],nargs='+',help='the epoch where optimizer reduce the learning rate')
        #GPU
        #parser.add_argument('--device',type=int,default=0,nargs='+',help='the indexes of GPUs for training or testing')
        #CPU
        parser.add_argument('--device', type=int, default=5, nargs='+', help='the indexes of GPUs for training or testing')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=False, help='use nesterov or not')
        parser.add_argument('--batch-size', type=int, default=256, help='training batch size')
        parser.add_argument('--test-batch-size', type=int, default=256, help='test batch size')
        parser.add_argument('--start-epoch',type=int,default=0,help='start training from which epoch')
        parser.add_argument( '--num-epoch',type=int,default=80,help='stop training in which epoch')
        parser.add_argument('--weight-decay', type=float,default=0.0005,help='weight decay for optimizer')
        parser.add_argument( '--lr-ratio', type=float,default=0.001, help='decay rate for learning rate')
        parser.add_argument('--lr-decay-rate',type=float, default=0.1,help='decay rate for learning rate')
        parser.add_argument('--warm_up_epoch', type=int, default=0)
        parser.add_argument('--loss-type', type=str, default='CE')
        return parser


