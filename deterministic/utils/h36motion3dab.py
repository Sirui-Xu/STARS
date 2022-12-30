from torch.utils.data import Dataset
import numpy as np
from utils import data_utils
from matplotlib import pyplot as plt
import torch

import os 

'''
adapted from
https://github.com/wei-mao-2019/HisRepItself/blob/master/utils/h36motion3d.py
'''


class Datasets(Dataset):

    def __init__(self, data_dir,input_n,output_n,skip_rate,actions=None,split=0):
        """
        :param path_to_data:
        :param actions:
        :param input_n:
        :param output_n:
        :param dct_used:
        :param split: 0 train, 1 testing, 2 validation
        :param sample_rate:
        """
        self.path_to_data = os.path.join(data_dir,'h3.6m/dataset')
        self.split = split
        self.in_n = input_n
        self.out_n = output_n
        self.sample_rate = skip_rate
        self.p3d = {}
        self.data_idx = []
        seq_len = self.in_n + self.out_n
        subs = np.array([[1, 6, 7, 8, 9], [11], [5]])
        # acts = data_utils.define_actions(actions)
        if actions is None:
            acts = ["walking", "eating", "smoking", "discussion", "directions",
                    "greeting", "phoning", "posing", "purchases", "sitting",
                    "sittingdown", "takingphoto", "waiting", "walkingdog",
                    "walkingtogether"]
        else:
            acts = actions
        # subs = np.array([[1], [11], [5]])
        # acts = ['walking']
        # 32 human3.6 joint name:
        joint_name = ["Hips", "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase", "Site", "LeftUpLeg", "LeftLeg",
                      "LeftFoot",
                      "LeftToeBase", "Site", "Spine", "Spine1", "Neck", "Head", "Site", "LeftShoulder", "LeftArm",
                      "LeftForeArm",
                      "LeftHand", "LeftHandThumb", "Site", "L_Wrist_End", "Site", "RightShoulder", "RightArm",
                      "RightForeArm",
                      "RightHand", "RightHandThumb", "Site", "R_Wrist_End", "Site"]

        SEED = 1234567890
        rng = np.random.RandomState(SEED)
        subs = subs[split]
        key = 0
        for subj in subs:
            for action_idx in np.arange(len(acts)):
                action = acts[action_idx]
                if self.split <= 1:
                    for subact in [1, 2]:  # subactions
                        print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, subact))
                        filename = '{0}/S{1}/{2}_{3}.txt'.format(self.path_to_data, subj, action, subact)
                        the_sequence = data_utils.readCSVasFloat(filename)
                        n, d = the_sequence.shape
                        the_sequence = np.array(the_sequence)

                        the_sequence = data_utils.expmap2xyz_global(the_sequence, rng)

                        for bias in range(self.sample_rate):
                            even_list = range(bias, n, self.sample_rate)
                            num_frames = len(even_list)
                            p3d = the_sequence[even_list]
                            self.p3d[key] = p3d.view(num_frames, -1).cpu().numpy()

                            valid_frames = np.arange(0, num_frames - seq_len + 1)
                            tmp_data_idx_1 = [key] * len(valid_frames)
                            tmp_data_idx_2 = list(valid_frames)
                            self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))
                            key += 1
                else:
                    print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, 1))
                    filename = '{0}/S{1}/{2}_{3}.txt'.format(self.path_to_data, subj, action, 1)
                    the_sequence1 = data_utils.readCSVasFloat(filename)
                    n, d = the_sequence1.shape
                    even_list = range(0, n, self.sample_rate)

                    num_frames1 = len(even_list)
                    the_sequence1 = np.array(the_sequence1)
                    # the_seq1 = torch.from_numpy(the_sequence1).float().cuda()
                    # the_seq1[:, 0:6] = 0
                    p3d1 = data_utils.expmap2xyz_global(the_sequence1, rng)
                    # self.p3d[(subj, action, 1)] = p3d1.view(num_frames1, -1).cpu().data.numpy()
                    self.p3d[key] = p3d1[even_list].view(num_frames1, -1).cpu().data.numpy()


                    print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, 2))
                    filename = '{0}/S{1}/{2}_{3}.txt'.format(self.path_to_data, subj, action, 2)
                    the_sequence2 = data_utils.readCSVasFloat(filename)
                    n, d = the_sequence2.shape
                    even_list = range(0, n, self.sample_rate)

                    num_frames2 = len(even_list)
                    the_sequence2 = np.array(the_sequence2)
                    # the_seq2 = torch.from_numpy(the_sequence2).float().cuda()
                    # the_seq2[:, 0:6] = 0
                    p3d2 = data_utils.expmap2xyz_global(the_sequence2, rng)

                    # self.p3d[(subj, action, 2)] = p3d2.view(num_frames2, -1).cpu().data.numpy()
                    self.p3d[key + 1] = p3d2[even_list].view(num_frames2, -1).cpu().data.numpy()

                    # print("action:{}".format(action))
                    # print("subact1:{}".format(num_frames1))
                    # print("subact2:{}".format(num_frames2))
                    fs_sel1, fs_sel2 = data_utils.find_indices_256(num_frames1, num_frames2, seq_len,
                                                                   input_n=self.in_n)

                    valid_frames = fs_sel1[:, 0]
                    tmp_data_idx_1 = [key] * len(valid_frames)
                    tmp_data_idx_2 = list(valid_frames)
                    self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))

                    valid_frames = fs_sel2[:, 0]
                    tmp_data_idx_1 = [key + 1] * len(valid_frames)
                    tmp_data_idx_2 = list(valid_frames)
                    self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))
                    key += 2

        # ignore constant joints and joints at same position with other joints
        joint_to_ignore = np.array([16, 20, 23, 24, 28, 31])
        dimensions_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
        self.dimensions_to_use = np.setdiff1d(np.arange(96), dimensions_to_ignore)

    def __len__(self):
        return np.shape(self.data_idx)[0]

    def __getitem__(self, item):
        key, start_frame = self.data_idx[item]
        fs = np.arange(start_frame, start_frame + self.in_n + self.out_n)
        return self.p3d[key][fs]
