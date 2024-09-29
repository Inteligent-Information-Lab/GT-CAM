import cv2
import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt



from matplotlib.backends.backend_agg import FigureCanvasAgg
import seaborn as sns
import mpl_toolkits.axisartist as axisartist
from mpl_toolkits.axisartist.axislines import Axes
time_scale_factor = [1,1,1,1,2,2,2,4,4,4]
matplotlib.use('Agg')

def plot_action(pose, edge, feature, save_dir, save_type):
    _, T, V, M = pose.shape
    feature = feature[0]
    fig = plt.figure(figsize=(10, 10))
    # ax = plt.axes(projection='3d')
    ax = fig.add_subplot(111, projection='3d')
    z_max = 0

    fea = ((feature - feature.min()) / (feature.max() - feature.min() + 1e-12))

    for t in range(0, T, 5):

        if t:
            z_max += pose[2, t - 1, :, :].max() * 1.5
        f = fea[t, :]

        for m in range(M):
            for i, j in edge:
                xi = pose[0, t, i, m]
                yi = pose[1, t, i, m]
                xj = pose[0, t, j, m]
                yj = pose[1, t, j, m]
                if not t:
                    zi = pose[2, t, i, m]
                    zj = pose[2, t, j, m]
                else:
                    zi = pose[2, t, i, m] + z_max
                    zj = pose[2, t, j, m] + z_max

                # ax.plot3D([xi, xj], [yi, yj], [zi, zj], 'gray')
                ax.plot3D([zi, zj], [xi, xj], [yi, yj], 'Blue', alpha=0.6)

            for v in range(V):
                
                x = pose[0, t, v, m]
                y = pose[1, t, v, m]
                z = pose[2, t, v, m] + z_max
                
                # if not np.isnan(f[v, m]):
                #   
                # if np.isnan(f[v, m]):
                #    f[v, m] =  0.5
                #    ax.scatter3D(z, x, y, color='red', s=20, alpha=f[v, m])
                ax.scatter3D(z, x, y, color='red', s=20, alpha=f[v, m])
    font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 23, }
    # x_min, x_max = pose[0,:,:,:].min(), pose[0,:,:,:].max()
    # y_min, y_max = pose[1,:,:,:].min(), pose[1,:,:,:].max()
    # z_min, z_max = pose[2,:,:,:].min(), z_max + pose[2, -1 ,:,:].max()
    # ax.set_xlabel(' time\n----->', fontsize=20)
    # ax.xaxis.labelpad = 16
    ax.set_xlabel('time')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    # x_ticks = np.arange(x_min-0.1, x_max+0.1, (x_max - x_min)/5)
    # y_ticks = np.arange(y_min-0.2, y_max+0.2, (y_max - y_min)/5)
    # z_ticks = np.arange(z_min-0.2, z_max+0.2, (z_max - z_min)/5)
    # ax.set_xticks(z_ticks)
    # ax.set_yticks(x_ticks)
    # ax.set_zticks(y_ticks)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # ax.view_init(elev=0, azim=0)
    ax.set_box_aspect((20, 6, 6))

    print("saveing")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig('{}/{}.svg'.format(save_dir, save_type))
    plt.savefig('{}/{}.pdf'.format(save_dir, save_type))
    plt.savefig('{}/{}.png'.format(save_dir, save_type))

    return