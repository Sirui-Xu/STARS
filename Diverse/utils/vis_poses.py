import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import subprocess

from utils.vis_util import draw_skeleton


def vis_traj(gt_pos, pre_pos=[], labels=[], comments='', title='',
             joint_to_plot=[2, 3, 7, 8, 15, 16, 17, 20, 21, 22], sub_title=[''] * 10):
    """
    @param gt_pos: seq_l x jn x 3
    @param pre_pos: n seq_l x jn x 3
    @param labels: n
    @param comments:
    @return:
    """
    assert len(pre_pos) == len(labels)
    assert len(gt_pos.shape) == 3
    assert len(sub_title) == len(joint_to_plot)
    seq_n, joint_n, _ = gt_pos.shape
    joint_name = np.array(["Hips", "rUpLeg", "rLeg", "rFoot", "rToeBase", "rToeSite", "lUpLeg", "lLeg",
                           "lFoot", "lToeBase", "lToeSite", "Spine", "Spine1", "Neck", "Head", "Site", "lShoulder",
                           "lArm", "lForeArm", "lHand", "lHandThumb", "lHandSite", "lWristEnd", "lWristSite",
                           "rShoulder", "rArm", "rForeArm", "rHand", "rHandThumb", "rHandSite",
                           "rWristEnd", "rWristSite"])
    joint_to_ignore = np.array([11, 16, 20, 23, 24, 28, 31])
    joint_used = np.setdiff1d(np.arange(32), joint_to_ignore)
    parents = np.array([-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 12, 15, 16, 17, 17, 12, 20, 21, 22, 22])
    is_right = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    # joint_to_plot = [2, 3, 7, 8, 16, 17, 21, 22]
    linestyles = ['-', '--', '-.', ':', '--', '-.']
    markers = ['o', 'v', '*', 'D', 's', 'p']
    colors = ['k', 'r', 'g', 'b', 'c', 'm']
    coord = ['x', 'y', 'z']
    fig = plt.figure(0, figsize=[6 * 3, 3 * len(joint_to_plot)])
    # plt.title(title, pad=1)
    fig.suptitle('{}_{}'.format(title.split('/')[-1].split('.pdf')[0], comments), fontsize=14)
    axs = fig.subplots(len(joint_to_plot), 3)
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.2, hspace=0.4)
    for i, jn in enumerate(joint_to_plot):
        axs[i, 1].set_title("{}_{}".format(joint_name[joint_used][jn], sub_title[i]), x=0.5, y=1)
        # axs[i, 1].legend()
        for j in range(3):
            axs[i, j].plot(np.arange(1, seq_n + 1), gt_pos[:, jn, j], linestyle=linestyles[0], marker=markers[0],
                           c=colors[0], label='GT')
            axs[i, j].set_xticks(np.arange(1, seq_n + 1))
            for k, pp in enumerate(pre_pos):
                axs[i, j].plot(np.arange(1, seq_n + 1), pp[:, jn, j], linestyle=linestyles[k + 1],
                               marker=markers[k + 1], c=colors[k + 1],
                               label=labels[k])
    axs[0, 1].legend()
    plt.savefig('{}'.format(title))
    # plt.savefig('test1.pdf'.format(title))
    plt.clf()
    plt.close()
    # plt.show()


def vis_poses(gt_pos, pre_pos=[], labels=[], comments='', title='', skeleton=None):
    """
    @param gt_pos: seq_l x jn x 3
    @param pre_pos: n seq_l x jn x 3
    @param labels: n
    @param comments:
    @return:
    """
    assert len(pre_pos) == len(labels)
    assert len(gt_pos.shape) == 3
    if not len(comments) == gt_pos.shape[0]:
        comments = [] * gt_pos.shape[0]
    seq_n, joint_n, _ = gt_pos.shape
    # joint_name = np.array(["Hips", "rUpLeg", "rLeg", "rFoot", "rToeBase", "rToeSite", "lUpLeg", "lLeg",
    #                        "lFoot", "lToeBase", "lToeSite", "Spine", "Spine1", "Neck", "Head", "Site", "lShoulder",
    #                        "lArm", "lForeArm", "lHand", "lHandThumb", "lHandSite", "lWristEnd", "lWristSite",
    #                        "rShoulder", "rArm", "rForeArm", "rHand", "rHandThumb", "rHandSite",
    #                        "rWristEnd", "rWristSite"])
    # joint_to_ignore = np.array([11, 16, 20, 23, 24, 28, 31])
    # joint_used = np.setdiff1d(np.arange(32), joint_to_ignore)
    # parents = np.array([-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 12, 15, 16, 17, 17, 12, 20, 21, 22, 22])
    # is_right = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    parents = skeleton._parents
    is_right = np.zeros_like(parents)
    is_right[skeleton._joints_right] = 1

    joint_to_plot = [2, 3, 15, 16, 17]
    linestyles = ['-', '--', '-.', ':', '--', '-.']
    markers = ['o', 'v', '*', 'D', 's', 'p']
    colors = ['k', 'r', 'g', 'b', 'c', 'm']
    coord = ['x', 'y', 'z']
    rot1 = np.array([[0.9239557, -0.0000000, -0.3824995],
                     [-0.1463059, 0.9239557, -0.3534126],
                     [0.3534126, 0.3824995, 0.8536941]])
    rot2 = np.array([[1.0000000, 0.0000000, 0.0000000],
                     [0.0000000, 0.9239557, -0.3824995],
                     [0.0000000, 0.3824995, 0.9239557]])
    rot3 = np.array([[0.9239557, 0.0000000, 0.3824995],
                     [0.1463059, 0.9239557, -0.3534126],
                     [-0.3534126, 0.3824995, 0.8536941]])
    rot = [rot3, rot2, rot1]
    trans = 4
    # axs = fig.subplots(1, len(rot))
    # get value scope
    pgt = []
    ppred = []
    scope = []
    pp_tmp = []
    for i, rr in enumerate(rot):
        pt = np.matmul(np.expand_dims(rr, axis=0), gt_pos.transpose([0, 2, 1])).transpose([0, 2, 1])
        pt[:, :, 2] = pt[:, :, 2] + trans
        pt = pt[:, :, :2] / pt[:, :, 2:]
        pgt.append(pt)
        pp_tmp.append(pt)
        ppred.append([])
        for j, pp in enumerate(pre_pos):
            pt = np.matmul(np.expand_dims(rr, axis=0), pp.transpose([0, 2, 1])).transpose([0, 2, 1])
            pt[:, :, 2] = pt[:, :, 2] + trans
            pt = pt[:, :, :2] / pt[:, :, 2:]
            pp_tmp.append(pt)
            ppred[i].append(pt)
    pp_tmp = np.vstack(pp_tmp)
    max_x = np.max(pp_tmp[:, :, 0])
    min_x = np.min(pp_tmp[:, :, 0])
    max_y = np.max(pp_tmp[:, :, 1])
    min_y = np.min(pp_tmp[:, :, 1])
    dx = (max_x - min_x) + 0.1 * (max_x - min_x)
    max_x = min_x + dx * len(rot)

    scope = [max_x, min_x, max_y, min_y]
    for jj in range(seq_n):
        fig = plt.figure(0, figsize=[3 * len(rot), 6])
        axs = fig.subplots(1, 1)
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.2, hspace=0.4)
        # for i in range(len(rot)):
        #     axs[i].set_axis_off()
        #     axs[i].axis('equal')
        #     axs[i].plot(scope[i][:2], scope[i][2:], c='w')
        axs.set_axis_off()
        axs.axis('equal')
        axs.plot(scope[:2], scope[2:], c='w')
        for i, rr in enumerate(rot):
            # pt = np.matmul(rr, gt_pos[jj].transpose([1, 0])).transpose([1, 0])
            # pt[:, 2] = pt[:, 2] + trans
            # pt = pt[:, :2] / pt[:, 2:]
            pgt[i][jj][:, 0] = pgt[i][jj][:, 0] + dx * i
            draw_skeleton(axs, pgt[i][jj], parents=parents, is_right=is_right, cols=[colors[0], colors[0]],
                          line_style=linestyles[0], label='GT' if i == 0 else None)
            for j, pp in enumerate(pre_pos):
                # pt = np.matmul(rr, pp[jj].transpose([1, 0])).transpose([1, 0])
                # pt[:, 2] = pt[:, 2] + trans
                # pt = pt[:, :2] / pt[:, 2:]
                ppred[i][j][jj][:, 0] = ppred[i][j][jj][:, 0] + dx * i
                draw_skeleton(axs, ppred[i][j][jj], parents=parents, is_right=is_right,
                              cols=[colors[j + 1], colors[j + 1]],
                              line_style=linestyles[j + 1], label=labels[j] if i == 0 else None)
        # axs[1].legend()
        # axs[1].set_title("f{}_{}".format(jj + 1, comments[jj]), x=0.5, y=1)
        # plt.savefig('{}/{}.jpg'.format(title, jj))
        axs.legend()
        axs.set_title("f{}_{}".format(jj + 1, comments[jj]), x=0.5, y=1)
        plt.savefig('{}/{}.jpg'.format(title, jj))
        # plt.show(block=False)
        # plt.pause(0.1)
        # for i in range(len(rot)):
        #     axs[i].clear()
        axs.clear()
        plt.clf()
        plt.close()
    # cmd = [
    #     'ffmpeg',
    #     '-i', vid_filename,
    #     f'{output_folder}/%06d.jpg',
    #     '-threads', '16'
    # ]
    #
    # print(' '.join(cmd))
    # try:
    #     subprocess.call(cmd)
    # except OSError:
    #     print('OSError')

    # # Set up formatting for the movie files
    # Writer = animation.writers['ffmpeg']
    # writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    # fig = plt.figure()
    # im = []
    # for jj in range(seq_n):
    #     img = plt.imread('{}_{}.jpg'.format(title.replace('.mp4', ''), jj))
    #     im_tmp = plt.imshow(img)
    #     im.append((im_tmp,))
    # im_ani = animation.ArtistAnimation(fig, im, interval=50, repeat_delay=3000,
    #                                    blit=True)
    # im_ani.save(title, writer=writer)
