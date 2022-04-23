from models import LinNF, model
import numpy as np

def get_model(cfg, dataset, model_type='h36m_naf'):
    specs = cfg.vae_specs
    traj_dim = dataset.traj_dim // 3

    if model_type == 'h36m' or model_type == 'humaneva':
        keep_joints = dataset.kept_joints[1:]
        # skeleton = dataset.skeleton
        # parents = skeleton.parents()[1:] - 1
        # joints_left = np.array(skeleton.joints_left()) - 1
        # joints_right = np.array(skeleton.joints_right()) - 1
        if model_type == 'h36m':
            parents=[-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 12,
                                          16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27, 28, 27, 30]
            joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23]
            joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31]
        elif model_type == 'humaneva':
            parents=[-1, 0, 1, 2, 3, 1, 5, 6, 0, 8, 9, 0, 11, 12, 1]
            joints_left=[2, 3, 4, 8, 9, 10]
            joints_right=[5, 6, 7, 11, 12, 13]

        pose_info = {'keep_joints': keep_joints, 
                     'parents': parents,
                     'joints_left': joints_left,
                     'joints_right': joints_right}
        print("Human pose information: ", pose_info)
        return model.Model(
                 n_pre = cfg.n_pre,
                 nk = [cfg.nk1, cfg.nk2],
                 input_channels = 3,
                 input_time_frame = cfg.t_his,
                 output_time_frame = cfg.t_pred,
                 st_gcnn_dropout = specs.get('dropout', 0.1),
                 joints_to_consider = traj_dim,
                 pose_info=pose_info), \
               LinNF.LinNF(data_dim=traj_dim * 3, num_layer=3)
    elif model_type == 'h36m_nf' or model_type == 'humaneva_nf':
        return LinNF.LinNF(data_dim=traj_dim, num_layer=cfg.nf_specs['num_flow_layer'])