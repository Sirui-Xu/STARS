import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def draw_skeleton(ax, kpts, parents=[], is_right=[], cols=["#3498db", "#e74c3c"], marker='o', line_style='-',
                  label=None):
    """

    :param kpts: joint_n*(3 or 2)
    :param parents:
    :return:
    """
    # ax = plt.subplot(111)
    joint_n, dims = kpts.shape
    # by default it is human 3.6m joints
    # [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 25, 26, 27, 17, 18, 19]
    # if len(parents) == 0:
    #     parents = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
    #     is_right = [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
    #     if cols == []:
    #         cols = ["#3498db", "#e74c3c"]
    # if parents == 'op':
    #     parents = [1, -1, 1, 2, 3, 1, 5, 6, 1, 8, 9, 1, 11, 12, 0, 0, 14, 15]
    # if parents == 'smpl':
    #     # parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
    #     parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
    #     # is_right = [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1]
    #     is_right = [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1]
    #     if cols == []:
    #         cols = ["#3498db", "#e74c3c"]
    # if parents == 'smpl_add':
    #     parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21, 15]
    #     is_right = [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    #     cols = ["#3498db", "#e74c3c"]
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    if dims > 2:
        ax.view_init(75, 90)
        ax.set_zlabel('Z Label')
    # if dims == 2:
    #     # idx_choosed = np.intersect1d(np.where(kpts[:, 0] > 0)[0], np.where(kpts[:, 1] > 0)[0])
    #     # ax.scatter(kpts[idx_choosed, 0], kpts[idx_choosed, 1], c=c, marker=marker, s=10)
    #     ax.scatter(kpts[:, 0], kpts[:, 1], c=cols[0], marker=marker, s=10)
    #     # for i in idx_choosed:
    #     #     ax.text(kpts[i, 0], kpts[i, 1], "{:d}".format(i), color=c)
    # else:
    #     ax.scatter(kpts[:, 0], kpts[:, 1], kpts[:, 2], c=cols[0], marker=marker, s=10)
    #     for i in range(kpts.shape[0]):
    #         ax.text(kpts[i, 0], kpts[i, 1], kpts[i, 2], "{:d}".format(i), color=cols[0])
    is_label = True
    for i in range(len(parents)):
        if parents[i] < 0:
            continue
        # if dims == 2:
        #     if not (parents[i] in idx_choosed and i in idx_choosed):
        #         continue

        if dims == 2:
            # ax.plot([kpts[parents[i], 0], kpts[i, 0]], [kpts[parents[i], 1], kpts[i, 1]], c=cols[is_right[i]],
            #         linestyle=line_style,
            #         alpha=0.5 if is_right[i] else 1, linewidth=3)
            if label is not None and is_label:
                ax.plot([kpts[parents[i], 0], kpts[i, 0]], [kpts[parents[i], 1], kpts[i, 1]], c=cols[is_right[i]],
                        linestyle=line_style,
                        alpha=1 if is_right[i] else 0.6, label=label)
                is_label = False
            else:
                ax.plot([kpts[parents[i], 0], kpts[i, 0]], [kpts[parents[i], 1], kpts[i, 1]], c=cols[is_right[i]],
                        linestyle=line_style,
                        alpha=1 if is_right[i] else 0.6)
        else:
            if label is not None and is_label:
                ax.plot([kpts[parents[i], 0], kpts[i, 0]], [kpts[parents[i], 1], kpts[i, 1]],
                        [kpts[parents[i], 2], kpts[i, 2]], linestyle=line_style, c=cols[is_right[i]],
                        alpha=1 if is_right[i] else 0.6, linewidth=3, label=label)
                is_label = False
            else:
                ax.plot([kpts[parents[i], 0], kpts[i, 0]], [kpts[parents[i], 1], kpts[i, 1]],
                        [kpts[parents[i], 2], kpts[i, 2]], linestyle=line_style, c=cols[is_right[i]],
                        alpha=1 if is_right[i] else 0.6, linewidth=3)

    return None