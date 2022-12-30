import torch
import numpy as np
import pickle


def h36m_valid_angle_check(p3d):
    """
    p3d: [bs,16,3] or [bs,48]
    """
    if p3d.shape[-1] == 48:
        p3d = p3d.reshape([p3d.shape[0], 16, 3])

    cos_func = lambda p1, p2: np.sum(p1 * p2, axis=1) / np.linalg.norm(p1, axis=1) / np.linalg.norm(p2, axis=1)
    data_all = p3d
    valid_cos = {}
    # Spine2LHip
    p1 = data_all[:, 3]
    p2 = data_all[:, 6]
    cos_gt_l = np.sum(p1 * p2, axis=1) / np.linalg.norm(p1, axis=1) / np.linalg.norm(p2, axis=1)
    # Spine2RHip
    p1 = data_all[:, 0]
    p2 = data_all[:, 6]
    cos_gt_r = np.sum(p1 * p2, axis=1) / np.linalg.norm(p1, axis=1) / np.linalg.norm(p2, axis=1)
    valid_cos['Spine2Hip'] = np.vstack((cos_gt_l, cos_gt_r))

    # LLeg2LeftHipPlane
    p0 = data_all[:, 3]
    p1 = data_all[:, 4] - data_all[:, 3]
    p2 = data_all[:, 5] - data_all[:, 4]
    n0 = np.cross(p0, p1)
    cos_gt_l = np.sum(n0 * p2, axis=1) / np.linalg.norm(n0, axis=1) / np.linalg.norm(p2, axis=1)
    # RLeg2RHipPlane
    p0 = data_all[:, 0]
    p1 = data_all[:, 1] - data_all[:, 0]
    p2 = data_all[:, 2] - data_all[:, 1]
    n0 = np.cross(p1, p0)
    cos_gt_r = np.sum(n0 * p2, axis=1) / np.linalg.norm(n0, axis=1) / np.linalg.norm(p2, axis=1)
    valid_cos['Leg2HipPlane'] = np.vstack((cos_gt_l, cos_gt_r))

    # Shoulder2Hip
    p1 = data_all[:, 10] - data_all[:, 7]
    p2 = data_all[:, 3]
    cos_gt_l = np.sum(p1 * p2, axis=1) / np.linalg.norm(p1, axis=1) / np.linalg.norm(p2, axis=1)
    p1 = data_all[:, 13] - data_all[:, 7]
    p2 = data_all[:, 0]
    cos_gt_r = np.sum(p1 * p2, axis=1) / np.linalg.norm(p1, axis=1) / np.linalg.norm(p2, axis=1)
    valid_cos['Shoulder2Hip'] = np.vstack((cos_gt_l, cos_gt_r))

    # Leg2ShoulderPlane
    p0 = data_all[:, 13]
    p1 = data_all[:, 10]
    p2 = data_all[:, 4]
    p3 = data_all[:, 1]
    n0 = np.cross(p0, p1)
    cos_gt_l = np.sum(n0 * p2, axis=1) / np.linalg.norm(n0, axis=1) / np.linalg.norm(p2, axis=1)
    cos_gt_r = np.sum(n0 * p3, axis=1) / np.linalg.norm(n0, axis=1) / np.linalg.norm(p3, axis=1)
    valid_cos['Leg2ShoulderPlane'] = np.vstack((cos_gt_l, cos_gt_r))

    # Shoulder2Shoulder
    p0 = data_all[:, 13] - data_all[:, 7]
    p1 = data_all[:, 10] - data_all[:, 7]
    cos_gt = np.sum(p0 * p1, axis=1) / np.linalg.norm(p0, axis=1) / np.linalg.norm(p1, axis=1)
    valid_cos['Shoulder2Shoulder'] = cos_gt

    # Neck2Spine
    p0 = data_all[:, 7] - data_all[:, 6]
    p1 = data_all[:, 6]
    cos_gt = np.sum(p0 * p1, axis=1) / np.linalg.norm(p0, axis=1) / np.linalg.norm(p1, axis=1)
    valid_cos['Neck2Spine'] = cos_gt

    # Spine2HipPlane1
    p0 = data_all[:, 3]
    p1 = data_all[:, 4] - data_all[:, 3]
    n0 = np.cross(p1, p0)
    p2 = data_all[:, 6]
    n1 = np.cross(p2, n0)
    cos_dir_l = np.sum(p0 * n1, axis=1) / np.linalg.norm(p0, axis=1) / np.linalg.norm(n1, axis=1)
    cos_gt_l = np.sum(n0 * p2, axis=1) / np.linalg.norm(n0, axis=1) / np.linalg.norm(p2, axis=1)
    p0 = data_all[:, 0]
    p1 = data_all[:, 1] - data_all[:, 0]
    n0 = np.cross(p0, p1)
    p2 = data_all[:, 6]
    n1 = np.cross(n0, p2)
    cos_dir_r = np.sum(p0 * n1, axis=1) / np.linalg.norm(p0, axis=1) / np.linalg.norm(n1, axis=1)
    cos_gt_r = np.sum(n0 * p2, axis=1) / np.linalg.norm(n0, axis=1) / np.linalg.norm(p2, axis=1)
    cos_gt_l1 = np.ones_like(cos_gt_l) * 0.5
    cos_gt_r1 = np.ones_like(cos_gt_r) * 0.5
    cos_gt_l1[cos_dir_l < 0] = cos_gt_l[cos_dir_l < 0]
    cos_gt_r1[cos_dir_r < 0] = cos_gt_r[cos_dir_r < 0]
    valid_cos['Spine2HipPlane1'] = np.vstack((cos_gt_l1, cos_gt_r1))

    # Spine2HipPlane2
    cos_gt_l2 = np.ones_like(cos_gt_l) * 0.5
    cos_gt_r2 = np.ones_like(cos_gt_r) * 0.5
    cos_gt_l2[cos_dir_l >= 0] = cos_gt_l[cos_dir_l >= 0]
    cos_gt_r2[cos_dir_r >= 0] = cos_gt_r[cos_dir_r >= 0]
    valid_cos['Spine2HipPlane2'] = np.vstack((cos_gt_l2, cos_gt_r2))

    # ShoulderPlane2HipPlane (25 Jan)
    p1 = data_all[:, 7] - data_all[:, 3]
    p2 = data_all[:, 7] - data_all[:, 0]
    p3 = data_all[:, 10]
    p4 = data_all[:, 13]
    n0 = np.cross(p2, p1)
    n1 = np.cross(p3, p4)
    cos_gt_l = np.sum(n0 * n1, axis=1) / np.linalg.norm(n0, axis=1) / np.linalg.norm(n1, axis=1)
    valid_cos['ShoulderPlane2HipPlane'] = cos_gt_l

    # Head2Neck
    p1 = data_all[:, 7] - data_all[:, 6]
    p2 = data_all[:, 8] - data_all[:, 7]
    cos_gt_l = np.sum(p1 * p2, axis=1) / np.linalg.norm(p1, axis=1) / np.linalg.norm(p2, axis=1)
    valid_cos['Head2Neck'] = cos_gt_l

    # Head2HeadTop
    p1 = data_all[:, 9] - data_all[:, 8]
    p2 = data_all[:, 8] - data_all[:, 7]
    cos_gt_l = np.sum(p1 * p2, axis=1) / np.linalg.norm(p1, axis=1) / np.linalg.norm(p2, axis=1)
    valid_cos['Head2HeadTop'] = cos_gt_l

    # HeadVerticalPlane2HipPlane
    p1 = data_all[:, 9] - data_all[:, 8]
    p2 = data_all[:, 8] - data_all[:, 7]
    n0 = np.cross(p1, p2)
    p3 = data_all[:, 9] - data_all[:, 7]
    n1 = np.cross(n0, p3)
    p4 = data_all[:, 7] - data_all[:, 0]
    p5 = data_all[:, 7] - data_all[:, 3]
    n2 = np.cross(p4, p5)
    cos_gt_l = cos_func(n1, n2)
    valid_cos['HeadVerticalPlane2HipPlane'] = cos_gt_l

    # Shoulder2Neck
    p1 = data_all[:, 10] - data_all[:, 7]
    p2 = data_all[:, 6] - data_all[:, 7]
    cos_gt_l = cos_func(p1, p2)
    p1 = data_all[:, 13] - data_all[:, 7]
    p2 = data_all[:, 6] - data_all[:, 7]
    cos_gt_r = cos_func(p1, p2)
    valid_cos['Shoulder2Neck'] = np.vstack((cos_gt_l, cos_gt_r))

    return valid_cos


def h36m_valid_angle_check_torch(p3d):
    """
    p3d: [bs,16,3] or [bs,48]
    """
    if p3d.shape[-1] == 48:
        p3d = p3d.reshape([p3d.shape[0], 16, 3])
    data_all = p3d
    cos_func = lambda p1, p2: torch.sum(p1 * p2, dim=1) / torch.norm(p1, dim=1) / torch.norm(p2, dim=1)

    valid_cos = {}
    # Spine2LHip
    p1 = data_all[:, 3]
    p2 = data_all[:, 6]
    cos_gt_l = torch.sum(p1 * p2, dim=1) / torch.norm(p1, dim=1) / torch.norm(p2, dim=1)
    # Spine2RHip
    p1 = data_all[:, 0]
    p2 = data_all[:, 6]
    cos_gt_r = torch.sum(p1 * p2, dim=1) / torch.norm(p1, dim=1) / torch.norm(p2, dim=1)
    valid_cos['Spine2Hip'] = torch.vstack((cos_gt_l, cos_gt_r))

    # LLeg2LeftHipPlane
    p0 = data_all[:, 3]
    p1 = data_all[:, 4] - data_all[:, 3]
    p2 = data_all[:, 5] - data_all[:, 4]
    n0 = torch.cross(p0, p1, dim=1)
    cos_gt_l = torch.sum(n0 * p2, dim=1) / torch.norm(n0, dim=1) / torch.norm(p2, dim=1)
    # RLeg2RHipPlane
    p0 = data_all[:, 0]
    p1 = data_all[:, 1] - data_all[:, 0]
    p2 = data_all[:, 2] - data_all[:, 1]
    n0 = torch.cross(p1, p0)
    cos_gt_r = torch.sum(n0 * p2, dim=1) / torch.norm(n0, dim=1) / torch.norm(p2, dim=1)
    valid_cos['Leg2HipPlane'] = torch.vstack((cos_gt_l, cos_gt_r))

    # Shoulder2Hip
    p1 = data_all[:, 10] - data_all[:, 7]
    p2 = data_all[:, 3]
    cos_gt_l = torch.sum(p1 * p2, dim=1) / torch.norm(p1, dim=1) / torch.norm(p2, dim=1)
    p1 = data_all[:, 13] - data_all[:, 7]
    p2 = data_all[:, 0]
    cos_gt_r = torch.sum(p1 * p2, dim=1) / torch.norm(p1, dim=1) / torch.norm(p2, dim=1)
    valid_cos['Shoulder2Hip'] = torch.vstack((cos_gt_l, cos_gt_r))

    # Leg2ShoulderPlane
    p0 = data_all[:, 13]
    p1 = data_all[:, 10]
    p2 = data_all[:, 4]
    p3 = data_all[:, 1]
    n0 = torch.cross(p0, p1)
    cos_gt_l = torch.sum(n0 * p2, dim=1) / torch.norm(n0, dim=1) / torch.norm(p2, dim=1)
    cos_gt_r = torch.sum(n0 * p3, dim=1) / torch.norm(n0, dim=1) / torch.norm(p3, dim=1)
    valid_cos['Leg2ShoulderPlane'] = torch.vstack((cos_gt_l, cos_gt_r))

    # Shoulder2Shoulder
    p0 = data_all[:, 13] - data_all[:, 7]
    p1 = data_all[:, 10] - data_all[:, 7]
    cos_gt = torch.sum(p0 * p1, dim=1) / torch.norm(p0, dim=1) / torch.norm(p1, dim=1)
    valid_cos['Shoulder2Shoulder'] = cos_gt

    # Neck2Spine
    p0 = data_all[:, 7] - data_all[:, 6]
    p1 = data_all[:, 6]
    cos_gt = torch.sum(p0 * p1, dim=1) / torch.norm(p0, dim=1) / torch.norm(p1, dim=1)
    valid_cos['Neck2Spine'] = cos_gt

    # Spine2HipPlane1
    p0 = data_all[:, 3]
    p1 = data_all[:, 4] - data_all[:, 3]
    n0 = torch.cross(p1, p0)
    p2 = data_all[:, 6]
    n1 = torch.cross(p2, n0)
    cos_dir_l = torch.sum(p0 * n1, dim=1) / torch.norm(p0, dim=1) / torch.norm(n1, dim=1)
    cos_gt_l = torch.sum(n0 * p2, dim=1) / torch.norm(n0, dim=1) / torch.norm(p2, dim=1)
    p0 = data_all[:, 0]
    p1 = data_all[:, 1] - data_all[:, 0]
    n0 = torch.cross(p0, p1)
    p2 = data_all[:, 6]
    n1 = torch.cross(n0, p2)
    cos_dir_r = torch.sum(p0 * n1, dim=1) / torch.norm(p0, dim=1) / torch.norm(n1, dim=1)
    cos_gt_r = torch.sum(n0 * p2, dim=1) / torch.norm(n0, dim=1) / torch.norm(p2, dim=1)
    cos_gt_l1 = cos_gt_l[cos_dir_l < 0]
    cos_gt_r1 = cos_gt_r[cos_dir_r < 0]
    valid_cos['Spine2HipPlane1'] = torch.hstack((cos_gt_l1, cos_gt_r1))

    # Spine2HipPlane2
    cos_gt_l2 = cos_gt_l[cos_dir_l >= 0]
    cos_gt_r2 = cos_gt_r[cos_dir_r >= 0]
    valid_cos['Spine2HipPlane2'] = torch.hstack((cos_gt_l2, cos_gt_r2))

    # ShoulderPlane2HipPlane (25 Jan)
    p1 = data_all[:, 7] - data_all[:, 3]
    p2 = data_all[:, 7] - data_all[:, 0]
    p3 = data_all[:, 10]
    p4 = data_all[:, 13]
    n0 = torch.cross(p2, p1)
    n1 = torch.cross(p3, p4)
    cos_gt_l = torch.sum(n0 * n1, dim=1) / torch.norm(n0, dim=1) / torch.norm(n1, dim=1)
    valid_cos['ShoulderPlane2HipPlane'] = cos_gt_l

    # Head2Neck
    p1 = data_all[:, 7] - data_all[:, 6]
    p2 = data_all[:, 8] - data_all[:, 7]
    cos_gt_l = torch.sum(p1 * p2, dim=1) / torch.norm(p1, dim=1) / torch.norm(p2, dim=1)
    valid_cos['Head2Neck'] = cos_gt_l

    # Head2HeadTop
    p1 = data_all[:, 9] - data_all[:, 8]
    p2 = data_all[:, 8] - data_all[:, 7]
    cos_gt_l = torch.sum(p1 * p2, dim=1) / torch.norm(p1, dim=1) / torch.norm(p2, dim=1)
    valid_cos['Head2HeadTop'] = cos_gt_l

    # HeadVerticalPlane2HipPlane
    p1 = data_all[:, 9] - data_all[:, 8]
    p2 = data_all[:, 8] - data_all[:, 7]
    n0 = torch.cross(p1, p2)
    p3 = data_all[:, 9] - data_all[:, 7]
    n1 = torch.cross(n0, p3)
    p4 = data_all[:, 7] - data_all[:, 0]
    p5 = data_all[:, 7] - data_all[:, 3]
    n2 = torch.cross(p4, p5)
    cos_gt_l = cos_func(n1, n2)
    valid_cos['HeadVerticalPlane2HipPlane'] = cos_gt_l

    # Shoulder2Neck
    p1 = data_all[:, 10] - data_all[:, 7]
    p2 = data_all[:, 6] - data_all[:, 7]
    cos_gt_l = cos_func(p1, p2)
    p1 = data_all[:, 13] - data_all[:, 7]
    p2 = data_all[:, 6] - data_all[:, 7]
    cos_gt_r = cos_func(p1, p2)
    valid_cos['Shoulder2Neck'] = torch.vstack((cos_gt_l, cos_gt_r))

    return valid_cos


def humaneva_valid_angle_check(p3d):
    """
    p3d: [bs,14,3] or [bs,42]
    """
    if p3d.shape[-1] == 42:
        p3d = p3d.reshape([p3d.shape[0], 14, 3])

    cos_func = lambda p1, p2: np.sum(p1 * p2, axis=1) / np.linalg.norm(p1, axis=1) / np.linalg.norm(p2, axis=1)
    data_all = p3d
    valid_cos = {}

    # LHip2RHip
    p1 = data_all[:, 7]
    p2 = data_all[:, 10]
    cos_gt_l = cos_func(p1, p2)
    valid_cos['LHip2RHip'] = cos_gt_l

    # Neck2HipPlane
    p1 = data_all[:, 7]
    p2 = data_all[:, 10]
    n0 = np.cross(p1, p2)
    p3 = data_all[:, 0]
    cos_gt_l = cos_func(n0, p3)
    valid_cos['Neck2HipPlane'] = cos_gt_l

    # Head2Neck
    p1 = data_all[:, 13] - data_all[:, 0]
    p2 = data_all[:, 0]
    cos_gt_l = cos_func(p1, p2)
    valid_cos['Head2Neck'] = cos_gt_l

    # Shoulder2Shoulder
    p1 = data_all[:, 1] - data_all[:, 0]
    p2 = data_all[:, 4] - data_all[:, 0]
    cos_gt_l = cos_func(p1, p2)
    valid_cos['Shoulder2Shoulder'] = cos_gt_l

    # ShoulderPlane2HipPlane
    p1 = data_all[:, 7] - data_all[:, 0]
    p2 = data_all[:, 10] - data_all[:, 0]
    n0 = np.cross(p1, p2)
    p3 = data_all[:, 1]
    p4 = data_all[:, 4]
    n1 = np.cross(p3, p4)
    cos_gt_l = cos_func(n0, n1)
    valid_cos['ShoulderPlane2HipPlane'] = cos_gt_l

    # Shoulder2Neck
    p1 = data_all[:, 1] - data_all[:, 0]
    p2 = data_all[:, 0]
    cos_gt_l = cos_func(p1, p2)
    p1 = data_all[:, 4] - data_all[:, 0]
    p2 = data_all[:, 0]
    cos_gt_r = cos_func(p1, p2)
    valid_cos['Shoulder2Neck'] = np.vstack((cos_gt_l, cos_gt_r))

    # Leg2HipPlane
    p1 = data_all[:, 7]
    p2 = data_all[:, 10]
    n0 = np.cross(p1, p2)
    p3 = data_all[:, 8] - data_all[:, 7]
    cos_gt_l = cos_func(n0, p3)
    p3 = data_all[:, 11] - data_all[:, 10]
    cos_gt_r = cos_func(n0, p3)
    valid_cos['Leg2HipPlane'] = np.vstack((cos_gt_l, cos_gt_r))

    # Foot2LegPlane
    p1 = data_all[:, 7] - data_all[:, 10]
    p2 = data_all[:, 11] - data_all[:, 10]
    n0 = np.cross(p1, p2)
    p3 = data_all[:, 12] - data_all[:, 11]
    cos_gt_l = cos_func(n0, p3)
    p1 = data_all[:, 7] - data_all[:, 10]
    p2 = data_all[:, 8] - data_all[:, 7]
    n0 = np.cross(p1, p2)
    p3 = data_all[:, 9] - data_all[:, 8]
    cos_gt_r = cos_func(n0, p3)
    valid_cos['Foot2LegPlane'] = np.vstack((cos_gt_l, cos_gt_r))

    # ForeArm2ShoulderPlane
    p1 = data_all[:, 4] - data_all[:, 0]
    p2 = data_all[:, 5] - data_all[:, 4]
    n0 = np.cross(p1, p2)
    p3 = data_all[:, 6] - data_all[:, 5]
    cos_gt_l = cos_func(n0, p3)
    p1 = data_all[:, 1] - data_all[:, 0]
    p2 = data_all[:, 2] - data_all[:, 1]
    n0 = np.cross(p2, p1)
    p3 = data_all[:, 3] - data_all[:, 2]
    cos_gt_r = cos_func(n0, p3)
    valid_cos['ForeArm2ShoulderPlane'] = np.vstack((cos_gt_l, cos_gt_r))

    return valid_cos


def humaneva_valid_angle_check_torch(p3d):
    """
    p3d: [bs,14,3] or [bs,42]
    """
    if p3d.shape[-1] == 42:
        p3d = p3d.reshape([p3d.shape[0], 14, 3])

    cos_func = lambda p1, p2: torch.sum(p1 * p2, dim=1) / torch.norm(p1, dim=1) / torch.norm(p2, dim=1)
    data_all = p3d
    valid_cos = {}

    # LHip2RHip
    p1 = data_all[:, 7]
    p2 = data_all[:, 10]
    cos_gt_l = cos_func(p1, p2)
    valid_cos['LHip2RHip'] = cos_gt_l

    # Neck2HipPlane
    p1 = data_all[:, 7]
    p2 = data_all[:, 10]
    n0 = torch.cross(p1, p2)
    p3 = data_all[:, 0]
    cos_gt_l = cos_func(n0, p3)
    valid_cos['Neck2HipPlane'] = cos_gt_l

    # Head2Neck
    p1 = data_all[:, 13] - data_all[:, 0]
    p2 = data_all[:, 0]
    cos_gt_l = cos_func(p1, p2)
    valid_cos['Head2Neck'] = cos_gt_l

    # Shoulder2Shoulder
    p1 = data_all[:, 1] - data_all[:, 0]
    p2 = data_all[:, 4] - data_all[:, 0]
    cos_gt_l = cos_func(p1, p2)
    valid_cos['Shoulder2Shoulder'] = cos_gt_l

    # ShoulderPlane2HipPlane
    p1 = data_all[:, 7] - data_all[:, 0]
    p2 = data_all[:, 10] - data_all[:, 0]
    n0 = torch.cross(p1, p2)
    p3 = data_all[:, 1]
    p4 = data_all[:, 4]
    n1 = torch.cross(p3, p4)
    cos_gt_l = cos_func(n0, n1)
    valid_cos['ShoulderPlane2HipPlane'] = cos_gt_l

    # Shoulder2Neck
    p1 = data_all[:, 1] - data_all[:, 0]
    p2 = data_all[:, 0]
    cos_gt_l = cos_func(p1, p2)
    p1 = data_all[:, 4] - data_all[:, 0]
    p2 = data_all[:, 0]
    cos_gt_r = cos_func(p1, p2)
    valid_cos['Shoulder2Neck'] = torch.vstack((cos_gt_l, cos_gt_r))

    # Leg2HipPlane
    p1 = data_all[:, 7]
    p2 = data_all[:, 10]
    n0 = torch.cross(p1, p2)
    p3 = data_all[:, 8] - data_all[:, 7]
    cos_gt_l = cos_func(n0, p3)
    p3 = data_all[:, 11] - data_all[:, 10]
    cos_gt_r = cos_func(n0, p3)
    valid_cos['Leg2HipPlane'] = torch.vstack((cos_gt_l, cos_gt_r))

    # Foot2LegPlane
    p1 = data_all[:, 7] - data_all[:, 10]
    p2 = data_all[:, 11] - data_all[:, 10]
    n0 = torch.cross(p1, p2)
    p3 = data_all[:, 12] - data_all[:, 11]
    cos_gt_l = cos_func(n0, p3)
    p1 = data_all[:, 7] - data_all[:, 10]
    p2 = data_all[:, 8] - data_all[:, 7]
    n0 = torch.cross(p1, p2)
    p3 = data_all[:, 9] - data_all[:, 8]
    cos_gt_r = cos_func(n0, p3)
    valid_cos['Foot2LegPlane'] = torch.vstack((cos_gt_l, cos_gt_r))

    # ForeArm2ShoulderPlane
    p1 = data_all[:, 4] - data_all[:, 0]
    p2 = data_all[:, 5] - data_all[:, 4]
    n0 = torch.cross(p1, p2)
    p3 = data_all[:, 6] - data_all[:, 5]
    cos_gt_l = cos_func(n0, p3)
    p1 = data_all[:, 1] - data_all[:, 0]
    p2 = data_all[:, 2] - data_all[:, 1]
    n0 = torch.cross(p2, p1)
    p3 = data_all[:, 3] - data_all[:, 2]
    cos_gt_r = cos_func(n0, p3)
    valid_cos['ForeArm2ShoulderPlane'] = torch.vstack((cos_gt_l, cos_gt_r))

    return valid_cos
