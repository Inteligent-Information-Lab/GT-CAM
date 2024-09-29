import copy
import numpy as np
import torch

unique_keys = ['left_hand', 'right_hand', 'left_leg', 'right_leg', 'trunk']

def totall(data,cutq,num1,num2,num3,num4,num5):
    cutq.append([value_numpy(data, num1, num2), value_numpy(data, num3, num4), value_numpy(data, num5, y='False')])
    return cutq

def data_reshape(data, mask):
    N, C, T, V, M = data.size()
    reshaped_data = torch.reshape(mask, [N, T, V, M, C])
    reshaped_data = reshaped_data.permute(0, 4, 1, 2, 3).contiguous()

    return reshaped_data

def value_numpy(data,x,y='False'):
    cx1 = np.zeros((25, 3))
    N, C, T, V, M = data.size()
    if y == 'False':
        if x == 'left_hand':
            cx1[[8, 9, 10, 11, 23, 24], :] = 1
        elif x == 'right_hand':
            cx1[[4, 5, 6, 7, 21, 22], :] = 1
        elif x == 'left_leg':
            cx1[[16, 17, 18, 19], :] = 1
        elif x == 'right_leg':
            cx1[[12, 13, 14, 15], :] = 1
        elif x == 'trunk':
            cx1[[0, 1, 2, 3, 20], :] = 1

    else:
        for i in range(5):
            if x == unique_keys[i] or y == unique_keys[i]:
                if unique_keys[i] == 'left_hand':
                    cx1[[8, 9, 10, 11, 23, 24], :] = 1
                elif unique_keys[i] == 'right_hand':
                    cx1[[4, 5, 6, 7, 21, 22], :] = 1
                elif unique_keys[i] == 'left_leg':
                    cx1[[16, 17, 18, 19], :] = 1
                elif unique_keys[i] == 'right_leg':
                    cx1[[12, 13, 14, 15], :] = 1
                elif unique_keys[i] == 'trunk':
                    cx1[[0, 1, 2, 3, 20], :] = 1

    cx1 = np.tile(cx1, (T * M, 1))
    cx1 = torch.from_numpy(cx1)
    cx1 = data_reshape(data, cx1)

    return cx1


def sum(unique_keys_q_3,cut_q,data,c1,c2,c3):
    unique_keys_q_3.append([(c1 + '_' + c2), c3[0] + '_' + c3[1], c3[2]])
    cut_q = totall(data, cut_q, c1, c2, c3[0], c3[1], c3[2])
    unique_keys_q_3.append([(c1 + '_' + c2), c3[0] + '_' + c3[2], c3[1]])
    cut_q = totall(data, cut_q, c1, c2, c3[0], c3[2], c3[1])

    # unique_keys_q_3.append([(c1 + '_' + c2), c3[1] + '_' + c3[0], c3[2]])
    # cut_q = totall(data, cut_q, c1, c2, c3[1], c3[0], c3[2])

    unique_keys_q_3.append([(c1 + '_' + c2), c3[1] + '_' + c3[2], c3[0]])
    cut_q = totall(data, cut_q, c1, c2, c3[1], c3[2], c3[0])
    unique_keys_q_3.append([(c1 + '_' + c2), c3[2] + '_' + c3[0], c3[1]])
    cut_q = totall(data, cut_q, c1, c2, c3[2], c3[0], c3[1])

    # unique_keys_q_3.append([(c1 + '_' + c2), c3[2] + '_' + c3[1], c3[0]])
    # cut_q = totall(data, cut_q, c1, c2, c3[2], c3[1], c3[0])

    # print('-------------')
    # print((c1 + '_' + c2), c3[0] + '_' + c3[1], c3[2])
    # print((c1 + '_' + c2), c3[0] + '_' + c3[2], c3[1])
    # print((c1 + '_' + c2), c3[1] + '_' + c3[0], c3[2])
    # print('-------------')

    return unique_keys_q_3,cut_q


def outofkeyw(data):
    unique_keys1 = []  # center
    unique_keys_q_3 = []
    # unique_keys_q_2 = []

    cut_q = []
    # q = 3
    for i in range(5):
        for y in range((i + 1), 5):
            unique_keys1.append([unique_keys[i] + '_' + unique_keys[y]])

    for i in range(unique_keys1.__len__()):
        # for y in range(5):
        c = str(unique_keys1[i]).replace("['", '').replace("']", '').split('_')

        if (len(c) == 4):
            c1 = c[0] + '_' + c[1]
            c2 = c[2] + '_' + c[3]
            c3 = copy.deepcopy(unique_keys)
            c3.remove(c1)
            c3.remove(c2)
            unique_keys_q_3,cut_q = sum(unique_keys_q_3,cut_q,data,c1,c2,c3)

        elif (len(c) == 3):
            if c[0] == 'trunk':
                c1 = c[0]
                c2 = c[1] + '_' + c[2]
                c3 = copy.deepcopy(unique_keys)
                c3.remove(c1)
                c3.remove(c2)
                unique_keys_q_3,cut_q = sum(unique_keys_q_3,cut_q,data,c1,c2,c3)

            else:
                c1 = c[0] + '_' + c[1]
                c2 = c[2]
                c3 = copy.deepcopy(unique_keys)

                c3.remove(c1)
                c3.remove(c2)

                unique_keys_q_3,cut_q = sum(unique_keys_q_3,cut_q,data,c1,c2,c3)

    # print(unique_keys_q_3)
    unique_keys_q_2=['left_hand,left_leg,right_hand,right_leg','trunk']
    # cut_q2 = [value_numpy(data, num1, num2), value_numpy(data, num3, num4), value_numpy(data, num5, y='False')]

    return unique_keys_q_2,unique_keys_q_3,cut_q

