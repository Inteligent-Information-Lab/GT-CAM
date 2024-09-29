import numpy as np
import os
import chardet
#读取骨架信息：
"""
numFrame:第一行，为样本的帧数。
  第一个循环：frameInfo：为每一帧的信息
  numBody：第二行，执行动作的人数。
    第二个循环：bodyInfo：为每一个人的信息：一行的信息分别是ID等
    numJoint：第三行，骨架节点数
        第三个循环：jointInfo：为节点的数据，我们只关注前三个为3D数据(x,y,z)
        
"""
def read_skeleton(file):
    with open(file, 'r') as f:

        skeleton_sequence = {}
        skeleton_sequence['numFrame'] = int(f.readline())
        skeleton_sequence['frameInfo'] = []
        for t in range(skeleton_sequence['numFrame']):
            frame_info = {}
            frame_info['numBody'] = int(f.readline())
            frame_info['bodyInfo'] = []
            for m in range(frame_info['numBody']):
                body_info = {}
                body_info_key = [
                    'bodyID', 'clipedEdges', 'handLeftConfidence',
                    'handLeftState', 'handRightConfidence', 'handRightState',
                    'isResticted', 'leanX', 'leanY', 'trackingState'
                ]
                body_info = {
                    k: float(v)
                    for k, v in zip(body_info_key, f.readline().split())
                }
                body_info['numJoint'] = int(f.readline())
                body_info['jointInfo'] = []
                for v in range(body_info['numJoint']):
                    joint_info_key = [
                        'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
                        'orientationW', 'orientationX', 'orientationY',
                        'orientationZ', 'trackingState'
                    ]
                    joint_info = {
                        k: float(v)
                        for k, v in zip(joint_info_key, f.readline().split())
                    }
                    body_info['jointInfo'].append(joint_info)
                frame_info['bodyInfo'].append(body_info)
            skeleton_sequence['frameInfo'].append(frame_info)
    return skeleton_sequence


def read_xyz(file, max_body=2, num_joint=25):
    seq_info = read_skeleton(file)
    data = np.zeros((3, seq_info['numFrame'], num_joint, max_body))
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    data[:, n, j, m] = [v['x'], v['y'], v['z']]
                else:
                    pass
    return data

def read_xyz_new(file, max_body=2, num_joint=25):
    seq_info = read_skeleton(file)
    num_body = seq_info['frameInfo'][0]['numBody']
    print("num of body", num_body)
    data = np.zeros((3, seq_info['numFrame'], num_joint, max_body))  # 修改这里的数组大小
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            if m < max_body:  # 检查第4个维度的索引是否越界
                for j, v in enumerate(b['jointInfo']):
                    if j < num_joint:  # 检查第3个维度的索引是否越界
                        data[:, n, j, m] = [v['x'], v['y'], v['z']]
                    else:
                        break  # 如果越界则结束当前循环
            else:
                break  # 如果越界则结束当前循环
    return data
