import copy
import datetime
import itertools
import math
import os
import argparse
import time
# import torchsnooper
from .processor import Processor
from torch.nn.functional import cross_entropy
from tools.utils.ntu_read_skeleton import read_xyz, read_xyz_new
from itertools import combinations, chain
# from keras.utils import to_categorical
import tools.utils as utils

import numpy as np
import torch
import torch.nn.functional as F
max_body = 2
num_joint = 25
max_frame = 300

drop_stride = 2
unique_keys = ['left_hand', 'right_hand', 'left_leg', 'right_leg', 'trunk']

#  记录文本
time_now = time.strftime("%Y%m%d-%H%M", time.localtime())
file_txt = open('result/' + time_now + '.txt', mode='a', encoding="utf-8")
# cd /media/szu/wanglei/Seagate_WL/st-gcn
# python main.py ShapleyCam --use_gpu False  --data_level test_set

class Explainer(Processor):
    def __init__(self, argv=None):
        super().__init__(argv)
        self.count = 0
        self.data_set_len = 0

    def start(self):
        # print(self.model)
        print(self.arg.cam_type)
        print("Using {} weights.".format(self.arg.valid))
        cams_dict = {'shapleycam': self.ShapleyCam}
        method = self.arg.cam_type.lower()
        start_time = time.time()
        out = self.process_cams(cams_dict[method])
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"代码运行时间: {elapsed_time} 秒")

    
    def cfg_hook(self):
        self.gradients, self.activations = [], []

        def backward_hook_fn(module, grad_input, grad_output):
            self.gradients.append(grad_output[0].detach())

        def forward_hook_fn(module, feat_in, feat_out):
            self.activations.append(feat_out.detach())

        def input_backward_hook_fn(module, feat_in, feat_out):
            self.input_grad = feat_out

        modules = list(self.model.modules())
        for module in modules:
            if isinstance(module, torch.nn.ReLU):
                module.register_forward_hook(forward_hook_fn)
                module.register_backward_hook(backward_hook_fn)
            if isinstance(module, torch.nn.Conv2d):
                module.register_backward_hook(input_backward_hook_fn)

        return

    def reset_hook_value(self):
        self.gradients, self.activations = [], []
        return

    def summary(self, ave_drop, incr, inse_auc, del_auc):
        if self.count == 0:
            self.ave_drop = ave_drop
            self.increase = incr
            self.inse_auc = inse_auc
            self.del_auc = del_auc

        else:
            self.ave_drop += ave_drop
            self.increase += incr
            self.inse_auc += inse_auc
            self.del_auc += del_auc

        self.count += 1

        # if self.count % 50 == 0:
        #     print("Average Drop:", self.ave_drop,
        #           ", Increase:", self.increase,
        #           ", Insertion AUC:", self.inse_auc,
        #           ", Deletion AUC:", self.del_auc,
        #           ", Data count:", self.count)
        if self.count % 50 == 0:
            # print("Cross Entropy:\n", self.loss_value)
            # for i in range(len(self.topk)):
            #     print("Top", self.topk[i], ":", self.top_k[:, i],
            #           "Batch size:", self.batch_size,
            #           "Count:", self.count,
            #           "Data count:", self.data_set_len)
            print("Average Drop:", self.ave_drop/self.count,
                  ", Increase:", self.increase/self.count,
                  ", Insersion AUC:", self.inse_auc/self.count,
                  ", Deletion AUC:", self.del_auc/self.count,
                  ", Data count:", self.count, 
                  ", cam:", self.arg.cam_type, 
                  ", Data count:", self.arg.valid,
                  ",五个联盟，基于score")
    def data_reshape(self, data, mask):
        N, C, T, V, M = data.size()
        reshaped_data = torch.reshape(mask, [N, T, V, M, C])
        reshaped_data = reshaped_data.permute(0, 4, 1, 2, 3).contiguous()

        return reshaped_data

    def reshape_saliency_map(self, saliency_map, data):
        bs, c, t, v, m = saliency_map.shape
        saliency_map = torch.nn.functional.interpolate(saliency_map, size=(data.size(2), v, m), mode='trilinear', align_corners=True)
        saliency_map = self.data_norm(saliency_map)
        return saliency_map

    def data_norm(self, data):
        bs, c, t, v, m = data.shape
        data_min = data.view(bs, -1).min(dim=-1, keepdim=True).values.view(bs, 1, 1, 1, 1)
        data_max = data.view(bs, -1).max(dim=-1, keepdim=True).values.view(bs, 1, 1, 1, 1)
        # denominator = torch.where(
        #     (data_max-data_min) != 0., data_max-data_min, torch.tensor(1.).double().to(self.dev))
        denominator = torch.where(
            (data_max-data_min) != 0., data_max-data_min, torch.ones_like(data_max).to(self.dev))

        return (data - data_min) / denominator

    def process_cams(self, cam_func):
        """
        :param cam_func: The interpretable method chosen：shapleyCam
        :return:
        """

        if self.arg.data_level == 'instance':
            label_name_path = './resource/ntu_skeleton/label_name.txt'
            with open(label_name_path) as f:
                label_name = f.readlines()
                label_name = [line.rstrip() for line in label_name]
                self.label_name = label_name

            self.cfg_hook()
            data_numpy, data,  label, probability, skeleton_name = self.lead_data()
            # print(label)
            # out = self.node_shapley_computer(data, label, 30)
            # print(out)
            file_txt.write(str(out))
            output_result_dir = '{}/{}'.format(self.arg.output_dir, skeleton_name)
            if not os.path.exists('{}/cam/'.format(output_result_dir)):
                os.makedirs('{}/cam/'.format(output_result_dir))

            saliency_map = cam_func(data, label)
            ave_drop, ave_increase, inse_auc, del_auc = self.evaluate(data, saliency_map)
            if self.arg.plot_action:
                utils.visualization_skeleton.plot_action(
                    data_numpy, self.model.graph.edge, saliency_map.squeeze(0).cpu().numpy(),
                    save_dir=output_result_dir, save_type=self.arg.cam_type)

            return ave_drop, ave_increase, inse_auc, del_auc

        if self.arg.data_level == 'test_set':

            count = 0
            ave_drop_all = 0
            ave_increase_all = 0
            inse_auc_all = 0
            del_auc_all = 0

            skeleton_file_path = './data/NTU-RGB-D/ntuall/'
            for root, dirs, files in os.walk(skeleton_file_path):

                for file in files:
                    print("=================================================================")
                    file_txt.write("=================================================================" + '\n')
                    file_txt.write(str(datetime.datetime.now()) + '\n')
                    file_txt.write(str(file) + '\n')
                    file_txt.write(str(self.arg.cam_type.lower()) + '\n')
                    file_txt.write(str(self.arg.valid) + '\n')
                    file_txt.write(str(self.arg.config) + '\n')
                    skeleton_file = skeleton_file_path + file
                    print(skeleton_file)
                    data_numpy = read_xyz(
                        skeleton_file, max_body=max_body, num_joint=num_joint)
                    self.model.eval().to(self.dev).double()
                    self.cfg_hook()
                    data = torch.from_numpy(data_numpy)
                    data = data.unsqueeze(0)
                    data = data.float().double()
                    print(np.shape(data))
                    data_model = data.to(self.dev).double()
                    with torch.no_grad():
                        out = self.model(data_model)
                        label = out.argmax(dim=1, keepdim=True)

                    output_result_dir = '{}/{}'.format(self.arg.output_dir, skeleton_file)
                    print("---------------------")
                    print(output_result_dir)
                    if not os.path.exists('{}/cam/'.format(output_result_dir)):
                        os.makedirs('{}/cam/'.format(output_result_dir))

                    saliency_map = cam_func(data, label)
                    ave_drop, ave_increase, inse_auc, del_auc = self.evaluate(data, saliency_map)
                    ave_drop_all += ave_drop
                    ave_increase_all += ave_increase
                    inse_auc_all += inse_auc
                    del_auc_all += del_auc
                    self.summary(ave_drop, ave_increase, inse_auc, del_auc)

                    if self.arg.plot_action:
                        utils.visualization_skeleton.plot_action(
                            data_numpy, self.model.graph.edge, saliency_map.squeeze(0).cpu().numpy(),
                            save_dir=output_result_dir, save_type=self.arg.cam_type)

                    count = count + 1
                    if count >= 2000:
                        break

            ave_drop_out = ave_drop_all / count
            ave_increase_out = ave_increase_all / count
            inse_auc_out = inse_auc_all / count
            del_auc_out = del_auc_all / count

            file_txt.write("================================END==============================" + '\n')
            file_txt.write("总的平均下降指标：" + str(ave_drop_out) + '\n')
            file_txt.write("总的平均上升指标：" + str(ave_increase_out) + '\n')
            file_txt.write("总的删除曲线指标：" + str(del_auc_out) + '\n')
            file_txt.write("总的插入曲线指标：" + str(inse_auc_out) + '\n')
            file_txt.write("================================END==============================" + '\n')

    def lead_data(self):
        skeleton_name = self.arg.skeleton
        action_class = int(
            skeleton_name[skeleton_name.find('A') + 1:skeleton_name.find('A') + 4]) - 1
        print("Processing skeleton:", skeleton_name)
        print("Skeleton class:", action_class)
        self.action_class = action_class

        # skeleton_file = 'D:\\s\\st-gcn\\data\\NTU-RGB-D\\nturgb+d_skeletons\\'
        skeleton_file = './data/NTU-RGB-D/nturgb+d_skeletons/'
        skeleton_file = skeleton_file + self.arg.skeleton + '.skeleton'
        data_numpy = read_xyz(
            skeleton_file, max_body=max_body, num_joint=num_joint)

        self.model.eval().double()
        data = torch.from_numpy(data_numpy)
        data = data.unsqueeze(0)
        data = data.float()
        data_model = data.to(self.dev).double()
        with torch.no_grad():
            out = self.model(data_model)
            label = out.argmax(dim=1, keepdim=True)
            probability = (torch.nn.functional.softmax(out[0], dim=-1))[label].item()
        
        return data_numpy, data,  label, probability, skeleton_name

    def cut_skeleton(self, data):
        """
        In order to make the skeleton data to five
        :param data: the skeleton data shape is (N, C, T, V, M)
        :return: five body:left_hand, right_hand, left_leg, right_leg, trunk
        """
        N, C, T, V, M = data.size()

        # 定义各个肢体的矩阵
        left_hand = np.zeros((25, 3))
        left_hand[[8, 9, 10, 11, 23, 24], :] = 1
        left_hands = np.tile(left_hand, (T*M, 1))
        left_hands = torch.from_numpy(left_hands)
        left_hands = self.data_reshape(data, left_hands)

        right_hand = np.zeros((25, 3))
        right_hand[[4, 5, 6, 7, 21, 22], :] = 1
        right_hands = np.tile(right_hand, (T*M, 1))
        right_hands = torch.from_numpy(right_hands)
        right_hands = self.data_reshape(data, right_hands)

        left_leg = np.zeros((25, 3))
        left_leg[[16, 17, 18, 19], :] = 1
        left_legs = np.tile(left_leg, (T*M, 1))
        left_legs = torch.from_numpy(left_legs)
        left_legs = self.data_reshape(data, left_legs)

        right_leg = np.zeros((25, 3))
        right_leg[[12, 13, 14, 15], :] = 1
        right_legs = np.tile(right_leg, (T*M, 1))
        right_legs = torch.from_numpy(right_legs)
        right_legs = self.data_reshape(data, right_legs)

        trunk = np.zeros((25, 3))
        trunk[[0, 1, 2, 3, 20], :] = 1
        trunks = np.tile(trunk, (T*M, 1))
        trunks = torch.from_numpy(trunks)
        trunks = self.data_reshape(data, trunks)

        return left_hands, right_hands, left_legs, right_legs, trunks

    def permutation_combination(self, data, label, left_hand, right_hand, left_leg, right_leg, trunk):
        """
        In order to permutation and combination the skeleton of human
        :param label:
        :param data: the translate of data ,shape is (N, T, V, M, C)
        :param left_hand: left hand of human skeleton: 9, 10, 11, 12, 24, 25
        :param right_hand: right hand of human skeleton: 5, 6, 7, 8 ,22, 23
        :param left_leg: left leg of human skeleton: 17, 18, 19, 20
        :param right_leg: right leg of human skeleton: 13, 14, 15, 16
        :param trunk: trunk of human skeleton: 1, 2, 3, 4, 21
        :return:  source_data: 32 permutations and combinations
        """
        source_data = {}
        self.model.eval().double()
        self.model.zero_grad()
        # <editor-fold desc="single. 单个一组 5 个">
        left_hand = (left_hand * data).float().to(self.dev).double()
        with torch.no_grad():
            left_hand_out = self.model(left_hand)
            probability_left_hand = (torch.nn.functional.softmax(left_hand_out[0], dim=-1))[label].item()
        source_data["left_hand"] = probability_left_hand

        right_hand = (right_hand * data).float().to(self.dev).double()
        with torch.no_grad():
            right_hand_out = self.model(left_hand)
            probability_right_hand = (torch.nn.functional.softmax(right_hand_out[0], dim=-1))[label].item()
        source_data["right_hand"] = probability_right_hand

        left_leg = (left_leg * data).float().to(self.dev).double()
        with torch.no_grad():
            left_leg_out = self.model(left_leg)
            probability_left_leg = (torch.nn.functional.softmax(left_leg_out[0], dim=-1))[label].item()
        source_data["left_leg"] = probability_left_leg

        right_leg = (right_leg * data).float().to(self.dev).double()
        with torch.no_grad():
            right_leg_out = self.model(right_leg)
            probability_right_leg = (torch.nn.functional.softmax(right_leg_out[0], dim=-1))[label].item()
        source_data["right_leg"] = probability_right_leg

        trunk = (trunk * data).float().to(self.dev).double()
        with torch.no_grad():
            trunk_out = self.model(right_leg)
            probability_trunk = (torch.nn.functional.softmax(trunk_out[0], dim=-1))[label].item()
        source_data["trunk"] = probability_trunk
        # </editor-fold>

        # <editor-fold desc="two. 两个一组 10个">
        data = data.to(self.dev)
        one_two = left_hand + right_hand
        one_two = (data * one_two).float().to(self.dev).double()
        with torch.no_grad():
            one_two_out = self.model(one_two)
            probability_one_two = (torch.nn.functional.softmax(one_two_out[0], dim=-1))[label].item()
        source_data["left_hand,right_hand"] = probability_one_two

        one_three = left_hand + left_leg
        one_three = (data * one_three).float().to(self.dev).double()
        with torch.no_grad():
            one_three_out = self.model(one_three)
            probability_one_three = (torch.nn.functional.softmax(one_three_out[0], dim=-1))[label].item()
        source_data["left_hand,left_leg"] = probability_one_three

        one_four = left_hand + right_leg
        one_four = (data * one_four).float().to(self.dev).double()
        with torch.no_grad():
            one_four_out = self.model(one_four)
            probability_one_four = (torch.nn.functional.softmax(one_four_out[0], dim=-1))[label].item()
        source_data["left_hand,right_leg"] = probability_one_four

        one_five = left_hand + trunk
        one_five = (data * one_five).float().to(self.dev).double()
        with torch.no_grad():
            one_five_out = self.model(one_five)
            probability_one_five = (torch.nn.functional.softmax(one_five_out[0], dim=-1))[label].item()
        source_data["left_hand,trunk"] = probability_one_five

        two_three = right_hand + left_leg
        two_three = (data * two_three).float().to(self.dev).double()
        with torch.no_grad():
            two_three_out = self.model(two_three)
            probability_two_three = (torch.nn.functional.softmax(two_three_out[0], dim=-1))[label].item()
        source_data["right_hand,left_leg"] = probability_two_three

        two_four = right_hand + right_leg
        two_four = (data * two_four).float().to(self.dev).double()
        with torch.no_grad():
            two_four_out = self.model(two_four)
            probability_two_four = (torch.nn.functional.softmax(two_four_out[0], dim=-1))[label].item()
        source_data["right_hand,right_leg"] = probability_two_four

        two_five = right_hand + trunk
        two_five = (data * two_five).float().to(self.dev).double()
        with torch.no_grad():
            two_five_out = self.model(two_five)
            probability_two_five = (torch.nn.functional.softmax(two_five_out[0], dim=-1))[label].item()
        source_data["right_hand,trunk"] = probability_two_five

        three_four = left_leg + right_leg
        three_four = (data * three_four).float().to(self.dev).double()
        with torch.no_grad():
            three_four_out = self.model(three_four)
            probability_three_four = (torch.nn.functional.softmax(three_four_out[0], dim=-1))[label].item()
        source_data["left_leg,right_leg"] = probability_three_four

        three_five = left_leg + trunk
        three_five = (data * three_five).float().to(self.dev).double()
        with torch.no_grad():
            three_five_out = self.model(three_five)
            probability_three_five = (torch.nn.functional.softmax(three_five_out[0], dim=-1))[label].item()
        source_data["left_leg,trunk"] = probability_three_five

        four_five = right_leg + trunk
        four_five = (data * four_five).float().to(self.dev).double()
        with torch.no_grad():
            four_five_out = self.model(four_five)
            probability_four_five = (torch.nn.functional.softmax(four_five_out[0], dim=-1))[label].item()
        source_data["right_leg,trunk"] = probability_four_five
        # </editor-fold>

        # <editor-fold desc="three. 三个一组 10个">
        one_two_three = one_two + left_leg
        one_two_three = (data * one_two_three).float().to(self.dev).double()
        with torch.no_grad():
            one_two_three_out = self.model(one_two_three)
            probability_one_two_three = (torch.nn.functional.softmax(one_two_three_out[0], dim=-1))[label].item()
        source_data["left_hand,right_hand,left_leg"] = probability_one_two_three

        one_two_four = one_two + right_leg
        one_two_four = (data * one_two_four).float().to(self.dev).double()
        with torch.no_grad():
            one_two_four_out = self.model(one_two_four)
            probability_one_two_four = (torch.nn.functional.softmax(one_two_four_out[0], dim=-1))[label].item()
        source_data["left_hand,right_hand,right_leg"] = probability_one_two_four

        one_two_five = one_two + trunk
        one_two_five = (data * one_two_five).float().to(self.dev).double()
        with torch.no_grad():
            one_two_five_out = self.model(one_two_five)
            probability_one_two_five = (torch.nn.functional.softmax(one_two_five_out[0], dim=-1))[label].item()
        source_data["left_hand,right_hand,trunk"] = probability_one_two_five

        one_three_four = one_three + right_leg
        one_three_four = (data * one_three_four).float().to(self.dev).double()
        with torch.no_grad():
            one_three_four_out = self.model(one_three_four)
            probability_one_three_four = (torch.nn.functional.softmax(one_three_four_out[0], dim=-1))[label].item()
        source_data["left_hand,left_leg,right_leg"] = probability_one_three_four

        one_three_five = one_three + trunk
        one_three_five = (data * one_three_five).float().to(self.dev).double()
        with torch.no_grad():
            one_three_five_out = self.model(one_three_five)
            probability_one_three_five = (torch.nn.functional.softmax(one_three_five_out[0], dim=-1))[label].item()
        source_data["left_hand,left_leg,trunk"] = probability_one_three_five

        one_four_five = one_four + trunk
        one_four_five = (data * one_four_five).float().to(self.dev).double()
        with torch.no_grad():
            one_four_five_out = self.model(one_four_five)
            probability_one_four_five = (torch.nn.functional.softmax(one_four_five_out[0], dim=-1))[label].item()
        source_data["left_hand,right_leg,trunk"] = probability_one_four_five

        two_three_four = two_three + right_leg
        two_three_four = (data * two_three_four).float().to(self.dev).double()
        with torch.no_grad():
            two_three_four_out = self.model(two_three_four)
            probability_two_three_four = (torch.nn.functional.softmax(two_three_four_out[0], dim=-1))[label].item()
        source_data["right_hand,left_leg,right_leg"] = probability_two_three_four

        two_three_five = two_three + trunk
        two_three_five = (data * two_three_five).float().to(self.dev).double()
        with torch.no_grad():
            two_three_five_out = self.model(two_three_five)
            probability_two_three_five = (torch.nn.functional.softmax(two_three_five_out[0], dim=-1))[label].item()
        source_data["right_hand,left_leg,trunk"] = probability_two_three_five

        two_four_five = two_four + trunk
        two_four_five = (data * two_four_five).float().to(self.dev).double()
        with torch.no_grad():
            two_four_five_out = self.model(two_four_five)
            probability_two_four_five = (torch.nn.functional.softmax(two_four_five_out[0], dim=-1))[label].item()
        source_data["right_hand,right_leg,trunk"] = probability_two_four_five

        three_four_five = three_four + trunk
        three_four_five = (data * three_four_five).float().to(self.dev).double()
        with torch.no_grad():
            three_four_five_out = self.model(three_four_five)
            probability_three_four_five = (torch.nn.functional.softmax(three_four_five_out[0], dim=-1))[label].item()
        source_data["left_leg,right_leg,trunk"] = probability_three_four_five
        # </editor-fold>

        # <editor-fold desc="four. 四个一组 5个">
        one_two_three_four = one_two_three + right_leg
        one_two_three_four = (data * one_two_three_four).float().to(self.dev).double()
        with torch.no_grad():
            one_two_three_four_out = self.model(one_two_three_four)
            probability_one_two_three_four = (torch.nn.functional.softmax(one_two_three_four_out[0], dim=-1))[label].item()
        source_data["left_hand,right_hand,left_leg,right_leg"] = probability_one_two_three_four

        one_two_three_five = one_two_three + trunk
        one_two_three_five = (data * one_two_three_five).float().to(self.dev).double()
        with torch.no_grad():
            one_two_three_five_out = self.model(one_two_three_five)
            probability_one_two_three_five = (torch.nn.functional.softmax(one_two_three_five_out[0], dim=-1))[label].item()
        source_data["left_hand,right_hand,left_leg,trunk"] = probability_one_two_three_five

        one_two_four_five = one_two_four + trunk
        one_two_four_five = (data * one_two_four_five).float().to(self.dev).double()
        with torch.no_grad():
            one_two_four_five_out = self.model(one_two_four_five)
            probability_one_two_four_five = (torch.nn.functional.softmax(one_two_four_five_out[0], dim=-1))[label].item()
        source_data["left_hand,right_hand,right_leg,trunk"] = probability_one_two_four_five

        one_three_four_five = one_three_four + trunk
        one_three_four_five = (data * one_three_four_five).float().to(self.dev).double()
        with torch.no_grad():
            one_three_four_five_out = self.model(one_three_four_five)
            probability_one_three_four_five = (torch.nn.functional.softmax(one_three_four_five_out[0], dim=-1))[label].item()
        source_data["left_hand,left_leg,right_leg,trunk"] = probability_one_three_four_five

        two_three_four_five = two_three_four + trunk
        two_three_four_five = (data * two_three_four_five).float().to(self.dev).double()
        with torch.no_grad():
            two_three_four_five_out = self.model(two_three_four_five)
            probability_two_three_four_five = (torch.nn.functional.softmax(two_three_four_five_out[0], dim=-1))[label].item()
        source_data["right_hand,left_leg,right_leg,trunk"] = probability_two_three_four_five
        # </editor-fold>

        # <editor-fold desc="five. 5个一组 1个">
        completion = data.float().to(self.dev).double()
        with torch.no_grad():
            completion_out = self.model(completion)
            label_new = completion_out.argmax(dim=1, keepdim=True)
            probability_completion = (torch.nn.functional.softmax(completion_out[0], dim=-1))[label].item()
        source_data["left_hand,right_hand,left_leg,right_leg,trunk"] = probability_completion
        # </editor-fold>

        return source_data
    
    def compute_shapley_values(self, unique_keys, source_data):
        """
        In order to Compute shapley values of each permutation
        :param unique_keys: the permutation of human skeleton
        :param source_data: the probability of each permutation
        :return: the shapley values of each permutation human skeleton
        """
        # 所有的联盟
        all_coalitions = [list(j) for i in range(len(unique_keys)) for j in itertools.combinations(unique_keys, r=i + 1)]

        # 联盟成员数
        n = len(unique_keys)
        shapley_value_list = []

        # 遍历每一个成员
        for k in unique_keys:

            # 包含k的联盟
            k_coalition = []
            # 剔除k后的联盟
            no_k_coalition = []
            # 夏普利值
            k_shapley_value = []

            # 找出所有包含k的联盟，并构造剔除k后的联盟
            for c in copy.deepcopy(all_coalitions):
                if k in c:
                    # 联盟贡献
                    k_coalition.append((copy.deepcopy(c), source_data[','.join(c)]))
                    # 剔除k后的贡献
                    c.remove(k)
                    if len(c) == 0:
                        no_k_coalition.append((c, 0))
                    else:
                        no_k_coalition.append((c, source_data[','.join(c)]))

            # file_txt.write("=================================================================" + '\n')
            file_txt.write(str(k) + "的联盟" + str(k_coalition) + '\n')
            # file_txt.write("=================================================================" + '\n')
            file_txt.write("剔除" + str(k) + "的联盟" + str(no_k_coalition) + '\n')
            # print('k的联盟:', k_coalition)
            # print('剔除k的联盟:', no_k_coalition)

            # 遍历包含k的联盟，并计算夏普利值
            shapley_value = 0
            for i in range(len(k_coalition)):
                s = len(k_coalition[i][0])
                # 成员k的边际贡献
                k_payoff = k_coalition[i][1] - no_k_coalition[i][1]
                # 联盟的权重系数
                k_weight = math.factorial(s-1) * math.factorial(n-s) / math.factorial(n)
                shapley_value += k_payoff * k_weight

            k_shapley_value.append((k, shapley_value))
            shapley_value_list.append(k_shapley_value)

            # 联盟的权重
            # file_txt.write("=================================================================" + '\n')
            file_txt.write(str(k) + "夏普利值:" + str(k_shapley_value) + '\n')
            # print('夏普利值：', k_shapley_value)

        print('各个部位的夏普利值：', shapley_value_list)
        # file_txt.write("=================================================================" + '\n')
        file_txt.write("各个部位的夏普利值:" + str(shapley_value_list) + '\n')

        return shapley_value_list

    def saliency_map(self, shapley_value, data):
        """
        In order to make the shapley value visualization and computer the evaluate
        :param shapley_value: each human skeleton shapley value
        :param data: the human skeleton data to get the each body of human
        :return: the evaluate and visualization of shapley value
        """
        left_hand, right_hand, left_leg, right_leg, trunk = self.cut_skeleton(data)
        shapley_value = np.array(shapley_value).squeeze(1)

        saliency_map = left_hand * float(shapley_value[0][1]) + right_hand * float(shapley_value[1][1]) + \
                       left_leg * float(shapley_value[2][1]) + right_leg * float(shapley_value[3][1]) + \
                       trunk * float(shapley_value[4][1])
        base_saliency_map= self.ScoreCam(data)
        base_saliency_map = base_saliency_map.cpu()
        saliency_map = torch.sum(saliency_map, dim=1, keepdim=True).float()
        saliency_map = saliency_map * saliency_map
        saliency_map = saliency_map * base_saliency_map

        return saliency_map

    def ShapleyCam(self, data, label):
        """
        In order to get the saliency_map of
        :param label:
        :param data: the original human skeleton data, shape is (N, C, T, V, M)
        :return:
        """
       
        left_hand, right_hand, left_leg, right_leg, trunk = self.cut_skeleton(data)
        source_data = self.permutation_combination(data, label, left_hand, right_hand, left_leg, right_leg, trunk)
        shapley_value = self.compute_shapley_values(unique_keys, source_data)
        saliency_map = self.saliency_map(shapley_value, data)

        return saliency_map

    def base_cam(self, data, action_class=None):
        """

        """
        self.model.eval()
        self.model.zero_grad()

        # action recognition
        data = data.to(self.dev)

        model_output = self.model(data)

        if action_class is None:
            action_class = model_output.argmax(dim=1, keepdim=True)

        one_hot = torch.zeros_like(model_output).to(self.dev)
        one_hot = one_hot.scatter_(1, action_class, 1)

        loss = torch.sum(one_hot * model_output)
        loss.backward(retain_graph=True)

        # 获取梯度
        gradients = self.gradients[0].detach()
        activations = self.activations[-1].detach()
        self.reset_hook_value()

        bs, c, t, v = gradients.shape  # 实际处理中bs和m已经合并
        gradients = gradients.view(-1, data.size(-1), c, t, v).permute(0, 2, 3, 4, 1).contiguous()
        activations = activations.view(-1, data.size(-1), c, t, v).permute(0, 2, 3, 4, 1).contiguous()
        bs, c, t, v, m = gradients.shape

        # 核心处理方法
        alpha = gradients.view(bs, c, -1).mean(2)
        weights = alpha.view(bs, c, 1, 1, 1)

        saliency_map = (weights * activations).sum(1, keepdim=True)
        saliency_map = torch.nn.functional.relu(saliency_map).detach()

        saliency_map = self.reshape_saliency_map(saliency_map, data)

        return saliency_map

    def evaluate(self, data, saliency_map):
        '''评价指标'''
        if saliency_map.dtype == torch.double:
            self.model = self.model.double()
            data = data.double()
        self.model.eval()
        self.model.zero_grad()

        data = data.cpu()
        cache = saliency_map.cpu()
        saliency_map = saliency_map.cpu()

        # 1. 计算ave drop和ave increase

        # np.percentile用来计算一组数的百分位数。
        threshold = np.percentile(cache, 50)
        # torch.where()函数的作用是按照一定的规则合并两个tensor类型。
        # torch.where(condition,a,b)其中 输入参数condition：条件限制,如果满足条件，则选择a，否则选择b作为输出。
        drop_data = torch.where(saliency_map.repeat(1, 3, 1, 1, 1) > threshold, data,
                                torch.zeros_like(data).cpu())
        data = data.to(self.dev)
        drop_data = drop_data.to(self.dev)

        with torch.no_grad():
            self.model.zero_grad()
            model_output = self.model(data)
            model_output = torch.nn.functional.softmax(model_output, dim=-1)
            self.reset_hook_value()

            self.model.zero_grad()
            drop_output = self.model(drop_data)
            drop_output = torch.nn.functional.softmax(drop_output, dim=-1)
            self.reset_hook_value()

        one_hot = torch.zeros(model_output.shape, dtype=torch.float32).to(self.dev)
        model_class = model_output.argmax(dim=1, keepdim=True)
        one_hot = one_hot.scatter_(1, model_class, 1)

        score = torch.sum(one_hot * model_output, dim=1)
        # print(score)
        drop_score = torch.sum(one_hot * drop_output, dim=1)
        # print(drop_score)

        average_drop = (torch.nn.functional.relu(score - drop_score) / score).sum().detach().cpu().numpy()
        average_increase = (score < drop_score).sum().detach().cpu().numpy()
        # print(average_drop)
        # print(average_increase)
        file_txt.write("average_drop:" + str(average_drop) + '\n')
        file_txt.write("average_increase:" + str(average_increase) + '\n')

        # 2.计算insertion 和 deletion
        drop_num = 100 // drop_stride
        threshold = [np.percentile(cache, i * drop_stride) for i in range(drop_num, 0, -1)]
        saliency_map = saliency_map.repeat(1, 3, 1, 1, 1)

        # insersion_list = torch.zeros((1, drop_num)).double().to(self.dev)
        insersion_list = []

        for drop_radio in range(drop_num):
            saliency_map = saliency_map.to(self.dev)
            drop_data = torch.where(saliency_map > threshold[drop_radio], data,
                                    torch.zeros_like(data)).detach()

            self.model.zero_grad()
            with torch.no_grad():
                drop_logit = self.model(drop_data)
                self.reset_hook_value()
                drop_logit = torch.nn.functional.softmax(drop_logit, dim=-1)
                drop_score = torch.sum(one_hot * drop_logit, dim=1)
            insersion_list.append(drop_score[0].detach().cpu().numpy())
            # insersion_list[:, drop_radio] = drop_score[0].detach()

        insersion_list = np.array(insersion_list)

        # insersion_list = insersion_list.cpu().numpy()
        # print(insersion_list)
        insersion_auc = insersion_list.sum() * drop_stride / 100
        # insersion_auc = insersion_auc.cpu().numpy()

        deletion_list = []

        for drop_radio in range(drop_num):
            drop_data = torch.where(saliency_map > threshold[drop_radio],
                                    torch.zeros_like(data), data).detach()

            self.model.zero_grad()
            with torch.no_grad():
                drop_logit = self.model(drop_data)
                self.reset_hook_value()
                drop_logit = torch.nn.functional.softmax(drop_logit, dim=-1)
                drop_score = torch.sum(one_hot * drop_logit, dim=1)
            deletion_list.append(drop_score[0].detach().cpu().numpy())

        deletion_list = np.array(deletion_list)
        deletion_auc = deletion_list.sum() * drop_stride / 100
        # deletion_auc = deletion_auc.cpu().numpy()

        file_txt.write("Insertion Curve:" + '\n' + str(insersion_list) + '\n')
        file_txt.write("Insertion AUC:" + str(insersion_auc) + '\n')
        file_txt.write("Deletion Curve:" + '\n' + str(deletion_list) + '\n')
        file_txt.write("Deletion AUC:" + str(deletion_auc) + '\n')
        if self.arg.data_level != 'test_set':
            print("Insertion Curve:\n", insersion_list)
            print("\nInsertion AUC:", insersion_auc)
            print("\nDeletion Curve:\n", deletion_list)
            print("\nDeletion AUC:", deletion_auc)

        return average_drop, average_increase, insersion_auc, deletion_auc

    def computer_relevance(self, index, m, data, label):
        bean_len = 25
        out = 0
        for cnt in range(m):
            perm = np.random.permutation(bean_len)
            perO = []

            for idx in perm:
                if idx != index:
                    perO.append(idx)
                else:
                    break

            without_index, with_index = self.change_data(index, perO, data)
            self.model.eval()
            self.model.zero_grad()
            without_index = without_index.float().to(self.dev)
            with_index = with_index.float().to(self.dev)
            with torch.no_grad():
                with_out = self.model(with_index)
                without_out = self.model(without_index)
                probability_with_out = (torch.nn.functional.softmax(with_out[0], dim=-1))[label].item()
                probability_without_out = (torch.nn.functional.softmax(without_out[0], dim=-1))[label].item()

            out += (probability_with_out - probability_without_out)

        return out / m

    def change_data(self, index, perO, data):
        N, C, T, V, M = data.size()
        bean_one = np.zeros((25, 3))
        bean_two = np.zeros((25, 3))

        bean_one[perO, :] = 1
        bean_one = np.tile(bean_one, (T * M, 1))
        bean_one = torch.from_numpy(bean_one)
        bean_one = self.data_reshape(data, bean_one)

        perO.append(index)
        bean_two[perO, :] = 1
        bean_two = np.tile(bean_two, (T * M, 1))
        bean_two = torch.from_numpy(bean_two)
        bean_two = self.data_reshape(data, bean_two)

        return bean_one, bean_two

    def node_shapley_computer(self, data, label, m):

        out = []
        for i in range(25):
            node_shapley = self.computer_relevance(i, m, data, label)
            out.append(node_shapley)

        return out

    @staticmethod
    def get_parser(add_help=False):
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Demo for Spatial Temporal Graph Convolution Network')

        # region arguments yapf: disable
        parser.add_argument('--skeleton',
                            default='S001C001P001R001A027',
                            help='Path to video')
        parser.add_argument('--openpose',
                            default=None,
                            help='Path to openpose')
        parser.add_argument('--plot_action',
                            default=False,
                            help='save action as image',
                            type=bool)
        parser.add_argument('--output_dir',
                            default='./data/demo_skeleton/result',
                            help='Path to save results')
        parser.add_argument('--cam_type',
                            default='shapleycam',
                            help='One of gradcam, gradcampp, smoothcam, \
                                ablation, scorecam, ada-gradcam, isgcam, l2-caf, shapleycam')
        parser.add_argument('--data_level',
                            default='test_set',
                            help='instance or test_set')
        # parser.add_argument('--valid',
        #                     default='xview',
        #                     help='One of xsub and xview')
        parser.add_argument('--valid',
                            default='csetup',
                            help='One of csub and csetup')
        parser.add_argument('--topk', type=int, default=[1, 5], nargs='+', help='which Top K accuracy will be shown')

        args = parser.parse_known_args(namespace=parent_parser)
        # print("mark", parent_parser.valid)
        # parser.set_defaults(
        #     config='./config/st_gcn/ntu-{}/test.yaml'.format(parent_parser.valid))
        parser.set_defaults(
            config='./config/st_gcn/ntu120-{}/test.yaml'.format(parent_parser.valid))
        parser.set_defaults(print_log=False)
        # endregion yapf: enable

        return parser
