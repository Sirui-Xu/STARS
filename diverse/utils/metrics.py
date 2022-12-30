import numpy as np
from scipy.spatial.distance import pdist

def mpjpe_error(pred, gt, *args): 
    indexs = np.zeros(pred.shape[0])
    sample_num, total_len, feature_size = pred.shape
    pred = pred.reshape([sample_num, total_len, feature_size//3, 3])[:, -1:, :, :] * 1000
    gt = gt.reshape([1, total_len, feature_size//3, 3])[:, -1:, :, :] * 1000
    dist = np.linalg.norm(pred - gt, axis=3).mean(axis=2).mean(axis=1)
    index = dist.argmin()
    indexs[index] += 1
    return dist[index], indexs

def compute_diversity(pred, *args):
    if pred.shape[0] == 1:
        return 0.0
    dist = pdist(pred.reshape(pred.shape[0], -1))
    diversity = dist.mean().item()
    return diversity, None


def compute_ade(pred, gt, *args):
    indexs = np.zeros(pred.shape[0])
    diff = pred - gt
    dist = np.linalg.norm(diff, axis=2).mean(axis=1)
    index = dist.argmin()
    indexs[index] += 1
    return dist[index], indexs


def compute_fde(pred, gt, *args):
    indexs = np.zeros(pred.shape[0])
    diff = pred - gt
    dist = np.linalg.norm(diff, axis=2)[:, -1]
    index = dist.argmin()
    indexs[index] += 1
    return dist[index], indexs

def compute_amse(pred, gt, *args):
    diff = pred - gt # sample_num * total_len * ((num_key-1)*3)
    dist = (diff*diff).sum() / diff.shape[0]
    return dist.mean(), None


def compute_fmse(pred, gt, *args):
    diff = pred[:, -1, :] - gt[:, -1, :] # sample_num * total_len * ((num_key-1)*3)
    dist = (diff*diff).sum() / diff.shape[0]
    return dist.mean(), None

def compute_mmade(pred, gt, gt_multi):
    gt_dist = []
    indexs = np.zeros(pred.shape[0])
    for gt_multi_i in gt_multi:
        dist, index = compute_ade(pred, gt_multi_i)
        gt_dist.append(dist)
        indexs += index
    gt_dist = np.array(gt_dist).mean()
    return gt_dist, indexs


def compute_mmfde(pred, gt, gt_multi):
    gt_dist = []
    indexs = np.zeros(pred.shape[0])
    for gt_multi_i in gt_multi:
        dist, index = compute_fde(pred, gt_multi_i)
        gt_dist.append(dist)
        indexs += index
    gt_dist = np.array(gt_dist).mean()
    return gt_dist, indexs
