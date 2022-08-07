import os
import sys
import pickle
import argparse
import time
from torch import maximum, optim
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.getcwd())
from utils import *
from motion_pred.utils.config import Config
from motion_pred.utils.dataset_h36m_multimodal import DatasetH36M
from motion_pred.utils.dataset_humaneva_multimodal import DatasetHumanEva
from motion_pred.utils.visualization import render_animation
from models.motion_pred import *
from utils import util, valid_angle_check
from utils.metrics import *
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
import random

def recon_loss(Y_g, Y, Y_mm, Y_hg, Y_h):
    stat = torch.zeros(Y_g.shape[2])

    diff = Y_g - Y.unsqueeze(2) # TBMV
    dist = diff.pow(2).sum(dim=-1).sum(dim=0) # BM
    value, indices = dist.min(dim=1)
    loss_recon_1 = value.mean()
    for i in range(cfg.nk):
        stat[i] = (indices == i).sum()
    stat /= stat.sum()

    diff = Y_hg - Y_h.unsqueeze(2) # TBMC
    loss_recon_2 = diff.pow(2).sum(dim=-1).sum(dim=0).mean()

    with torch.no_grad():
        ade = torch.norm(diff, dim=-1).mean(dim=0).min(dim=1)[0].mean()

    diff = Y_g[:, :, :, None, :] - Y_mm[:, :, None, :, :]
    mask = Y_mm.abs().sum(-1).sum(0) > 1e-6
    dist = diff.pow(2)
    with torch.no_grad():
        zeros = torch.zeros_like(dist, requires_grad=False).to(dist.device)
        const = dist.max() - dist.min()
        for i in range(indices.shape[0]):
            zeros[:, i, indices[i], :, :] = const + 1
            # Y_g[:, i, indices[i], :] = Y[:, i, :]
        dist += zeros
    dist = dist.sum(dim=-1).sum(dim=0)
    value_2, indices_2 = dist.min(dim=1)
    loss_recon_multi = value_2[mask].mean()
    if torch.isnan(loss_recon_multi):
        loss_recon_multi = torch.zeros_like(loss_recon_1)

    mask = torch.tril(torch.ones([cfg.nk, cfg.nk], device=device)) == 0
    yt = Y_g.reshape([-1, cfg.nk, Y_g.shape[3]]).contiguous()
    pdist = torch.cdist(yt, yt, p=1)[:, mask]

    return loss_recon_1, loss_recon_2, loss_recon_multi, ade, stat, (-pdist / 100).exp().mean()


def angle_loss(y):
    ang_names = list(valid_ang.keys())
    y = y.reshape([-1, y.shape[-1]])
    ang_cos = valid_angle_check.h36m_valid_angle_check_torch(
        y) if cfg.dataset == 'h36m' else valid_angle_check.humaneva_valid_angle_check_torch(y)
    loss = tensor(0, dtype=dtype, device=device)
    b = 1
    for an in ang_names:
        lower_bound = valid_ang[an][0]
        if lower_bound >= -0.98:
            # loss += torch.exp(-b * (ang_cos[an] - lower_bound)).mean()
            if torch.any(ang_cos[an] < lower_bound):
                # loss += b * torch.exp(-(ang_cos[an][ang_cos[an] < lower_bound] - lower_bound)).mean()
                loss += (ang_cos[an][ang_cos[an] < lower_bound] - lower_bound).pow(2).mean()
        upper_bound = valid_ang[an][1]
        if upper_bound <= 0.98:
            # loss += torch.exp(b * (ang_cos[an] - upper_bound)).mean()
            if torch.any(ang_cos[an] > upper_bound):
                # loss += b * torch.exp(ang_cos[an][ang_cos[an] > upper_bound] - upper_bound).mean()
                loss += (ang_cos[an][ang_cos[an] > upper_bound] - upper_bound).pow(2).mean()
    return loss


def loss_function(traj_est, traj, traj_multimodal, prior_lkh, prior_logdetjac, _lambda, mu, logvar):
    lambdas = cfg.nf_specs['lambdas']
    nj = dataset.traj_dim // 3

    Y_g = traj_est[t_his:] # T B M V
    Y = traj[t_his:]
    Y_multimodal = traj_multimodal[t_his:]

    RECON, RECON_2, RECON_mm, ade, stat, JL = recon_loss(Y_g, Y, Y_multimodal, traj_est[:t_his], traj[:t_his])

    # maintain limb length
    parent = dataset.skeleton.parents()
    tmp = traj[0].reshape([cfg.batch_size, nj, 3])
    pgt = torch.zeros([cfg.batch_size, nj + 1, 3], dtype=dtype, device=device)
    pgt[:, 1:] = tmp
    limbgt = torch.norm(pgt[:, 1:] - pgt[:, parent[1:]], dim=2)[None, :, None, :]
    tmp = traj_est.reshape([-1, cfg.batch_size, cfg.nk, nj, 3])
    pest = torch.zeros([tmp.shape[0], cfg.batch_size, cfg.nk, nj + 1, 3], dtype=dtype, device=device)
    pest[:, :, :, 1:] = tmp
    limbest = torch.norm(pest[:, :, :, 1:] - pest[:, :, :, parent[1:]], dim=4)
    loss_limb = torch.mean((limbgt - limbest).pow(2).sum(dim=3))

    # angle loss
    loss_ang = angle_loss(Y_g)
    if _lambda < 0.1:
        _lambda *= 10
    else:
        _lambda = 1
    loss_r = loss_limb * lambdas[1] + JL * lambdas[3] * _lambda + RECON * lambdas[4] + RECON_mm * lambdas[5] \
             - prior_lkh.mean() * lambdas[6] + RECON_2 * lambdas[7]# - prior_logdetjac.mean() * lambdas[7]
    
    KLD = lambdas[0] * -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / Y.shape[1]

    if loss_ang > 0:
        loss_r += loss_ang * lambdas[8]
    return loss_r, np.array([loss_r.item(), loss_limb.item(), loss_ang.item(),
                             JL.item(), RECON.item(), RECON_2.item(), RECON_mm.item(), ade.item(),
                             prior_lkh.mean().item(), prior_logdetjac.mean().item(), KLD.item()]), stat


def train(epoch, stats):
    model.train()
    t_s = time.time()
    train_losses = 0
    train_grad = 0
    train_grad_d = 0
    total_num_sample = 0
    n_modality = 10
    loss_names = ['LOSS', 'loss_limb', 'loss_ang', 'loss_DIV',
                  'RECON', 'RECON_2', 'RECON_multi', "ADE", 'p(z)', 'logdet', 'KLD']
    generator = dataset.sampling_generator(num_samples=cfg.num_vae_data_sample, batch_size=cfg.batch_size,
                                           n_modality=n_modality)
    prior = torch.distributions.Normal(torch.tensor(0, dtype=dtype, device=device),
                                       torch.tensor(1, dtype=dtype, device=device))
    
    for traj_np, traj_multimodal_np in tqdm(generator):
        with torch.no_grad():
            bs, _, nj, _ = traj_np[..., 1:, :].shape
            traj_np = traj_np[..., 1:, :].reshape(traj_np.shape[0], traj_np.shape[1], -1) # n t vc
            traj = tensor(traj_np, device=device, dtype=dtype).permute(1, 0, 2).contiguous() # t n vc 
            X = traj[:t_his]
            Y = traj[t_his:]

            traj_multimodal_np = traj_multimodal_np[..., 1:, :]  # [bs, modality, seqn, jn, 3]
            traj_multimodal_np = traj_multimodal_np.reshape([bs, n_modality, t_his + t_pred, -1]).transpose(
                [2, 0, 1, 3])
            traj_multimodal = tensor(traj_multimodal_np, device=device, dtype=dtype)  # .permute(0, 2, 1).contiguous()
        
        traj_est, mu, logvar = model(X, Y)

        # to save computation
        ran = np.random.uniform()
        if ran > 0.67:
            traj_tmp = traj_est[t_his::3].reshape([-1, traj_est.shape[-1] // 3, 3])
            tmp = torch.zeros_like(traj_tmp[:, :1, :])
            traj_tmp = torch.cat([tmp, traj_tmp], dim=1)
            traj_tmp = util.absolute2relative_torch(traj_tmp, parents=dataset.skeleton.parents()).reshape(
                [-1, traj_est.shape[-1]])
        elif ran > 0.33:
            traj_tmp = traj_est[t_his + 1::3].reshape([-1, traj_est.shape[-1] // 3, 3])
            tmp = torch.zeros_like(traj_tmp[:, :1, :])
            traj_tmp = torch.cat([tmp, traj_tmp], dim=1)
            traj_tmp = util.absolute2relative_torch(traj_tmp, parents=dataset.skeleton.parents()).reshape(
                [-1, traj_est.shape[-1]])
        else:
            traj_tmp = traj_est[t_his + 2::3].reshape([-1, traj_est.shape[-1] // 3, 3])
            tmp = torch.zeros_like(traj_tmp[:, :1, :])
            traj_tmp = torch.cat([tmp, traj_tmp], dim=1)
            traj_tmp = util.absolute2relative_torch(traj_tmp, parents=dataset.skeleton.parents()).reshape(
                [-1, traj_est.shape[-1]])
        z, prior_logdetjac = pose_prior(traj_tmp)

        prior_lkh = prior.log_prob(z).sum(dim=-1)
        # prior_logdetjac = log_det_jacobian.sum(dim=2)

        loss, losses, stat = loss_function(traj_est, traj, traj_multimodal, prior_lkh, prior_logdetjac, epoch / cfg.num_vae_epoch, mu, logvar)
        stats += stat
        # if torch.isinf(loss):
        #     print(1)
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(list(model.parameters()), max_norm=100)
        train_grad += grad_norm
        optimizer.step()
        train_losses += losses
        total_num_sample += 1
        # print(torch.cuda.memory_allocated()/1024/1024)
        del loss, z, traj_est#inp, xt, traj_est
        # print(torch.cuda.memory_allocated())

    scheduler.step()
    # dt = time.time() - t_s
    train_losses /= total_num_sample
    lr = optimizer.param_groups[0]['lr']
    losses_str = ' '.join(['{}: {:.4f}'.format(x, y) for x, y in zip(loss_names, train_losses)])

    # average cost of log time 20s
    tb_logger.add_scalar('train_grad', train_grad / total_num_sample, epoch)
    for name, loss in zip(loss_names, train_losses):
        tb_logger.add_scalars(name, {'train': loss}, epoch)

    logger.info('====> Epoch: {} Time: {:.2f} {} lr: {:.5f} branch_stats: {}'.format(epoch, time.time() - t_s, losses_str, lr, stats))
    return stats


def get_multimodal_gt(dataset_test):
    all_data = []
    data_gen = dataset_test.iter_generator(step=cfg.t_his)
    for data, _ in tqdm(data_gen):
        # print(data.shape)
        data = data[..., 1:, :].reshape(data.shape[0], data.shape[1], -1)
        all_data.append(data)
    all_data = np.concatenate(all_data, axis=0)
    all_start_pose = all_data[:, t_his - 1, :]
    pd = squareform(pdist(all_start_pose))
    traj_gt_arr = []
    num_mult = []
    for i in range(pd.shape[0]):
        ind = np.nonzero(pd[i] < args.multimodal_threshold)
        traj_gt_arr.append(all_data[ind][:, t_his:, :])
        num_mult.append(len(ind[0]))
    # np.savez_compressed('./data/data_3d_h36m_test.npz',data=all_data)
    # np.savez_compressed('./data/data_3d_humaneva15_test.npz',data=all_data)
    num_mult = np.array(num_mult)
    logger.info('')
    logger.info('')
    logger.info('=' * 80)
    logger.info(f'#1 future: {len(np.where(num_mult == 1)[0])}/{pd.shape[0]}')
    logger.info(f'#<10 future: {len(np.where(num_mult < 10)[0])}/{pd.shape[0]}')
    return traj_gt_arr

def get_prediction(data, model, sample_num, num_seeds=1, concat_hist=True):
    # 1 * total_len * num_key * 3
    traj_np = data[..., 1:, :].reshape(data.shape[0], data.shape[1], -1)
    # 1 * total_len * ((num_key-1)*3)
    traj = tensor(traj_np, device=device, dtype=dtype).permute(1, 0, 2).contiguous()
    # total_len * 1 * ((num_key-1)*3)
    X = traj[:t_his]
    Y_gt = traj[t_his:]
    X = X.repeat((1, sample_num * num_seeds, 1))
    Y_gt = Y_gt.repeat((1, sample_num * num_seeds, 1))
    # total_len * batch_size * feature_size
    Y, _, _ = model(X, Y_gt)
    Y = Y[t_his:]
    if concat_hist:
        # X = X.repeat((1, cfg.nk * sample_num * num_seeds, 1))
        # T B 1 V
        X = X.unsqueeze(2).repeat(1, sample_num * num_seeds, cfg.nk, 1)
        Y = torch.cat((X, Y), dim=0)
    # total_len * batch_size * feature_size
    Y = Y.squeeze(1).permute(1, 0, 2).contiguous().cpu().numpy()
    # batch_size * total_len * feature_size
    if Y.shape[0] > 1:
        Y = Y.reshape(-1, cfg.nk * sample_num, Y.shape[-2], Y.shape[-1])
    else:
        Y = Y[None, ...]
    # num_seeds * sample_num * total_len * feature_size
    return Y


def test(model, epoch):
    stats_func = {'Diversity': compute_diversity, 'AMSE': compute_amse, 'FMSE': compute_fmse, 'ADE': compute_ade,
                  'FDE': compute_fde, 'MMADE': compute_mmade, 'MMFDE': compute_mmfde, 'MPJPE': mpjpe_error}
    stats_names = list(stats_func.keys())
    stats_names.extend(['ADE_stat', 'FDE_stat', 'MMADE_stat', 'MMFDE_stat', 'MPJPE_stat'])
    stats_meter = {x: AverageMeter() for x in stats_names}

    data_gen = dataset_test.iter_generator(step=cfg.t_his)
    num_samples = 0
    num_seeds = 1
    
    for i, (data, _) in tqdm(enumerate(data_gen)):
        if args.mode == 'train' and (i >= 500 and (epoch + 1) % 50 != 0 and (epoch + 1) < cfg.num_vae_epoch - 100):
            break
        num_samples += 1
        gt = data[..., 1:, :].reshape(data.shape[0], data.shape[1], -1)[:, t_his:, :]
        gt_multi = traj_gt_arr[i]
        if gt_multi.shape[0] == 1:
            continue
        pred = get_prediction(data, model, sample_num=1, num_seeds=num_seeds, concat_hist=False)
        for stats in stats_names[:8]:
            val = 0
            branches = 0
            for pred_i in pred:
                # sample_num * total_len * ((num_key-1)*3), 1 * total_len * ((num_key-1)*3)
                v = stats_func[stats](pred_i, gt, gt_multi)
                val += v[0] / num_seeds
                if stats_func[stats](pred_i, gt, gt_multi)[1] is not None:
                    branches += v[1] / num_seeds
            stats_meter[stats].update(val)
            if type(branches) is not int:
                stats_meter[stats + '_stat'].update(branches)

    logger.info('=' * 80)
    for stats in stats_names:
        str_stats = f'Total {stats}: ' + f'{stats_meter[stats].avg}'
        logger.info(str_stats)
    logger.info('=' * 80)


def visualize():
    def denomarlize(*data):
        out = []
        for x in data:
            x = x * dataset.std + dataset.mean
            out.append(x)
        return out

    def post_process(pred, data):
        pred = pred.reshape(pred.shape[0], pred.shape[1], -1, 3)
        if cfg.normalize_data:
            pred = denomarlize(pred)
        pred = np.concatenate((np.tile(data[..., :1, :], (pred.shape[0], 1, 1, 1)), pred), axis=2)
        pred[..., :1, :] = 0
        return pred

    def pose_generator():

        while True:
            data, data_multimodal, action = dataset_test.sample(n_modality=10)
            gt = data[0].copy()
            gt[:, :1, :] = 0

            poses = {'action': action, 'context': gt, 'gt': gt}
            with torch.no_grad():
                pred = get_prediction(data, model, 1)[0]
                pred = post_process(pred, data)
                for i in range(pred.shape[0]):
                    poses[f'{i}'] = pred[i]

            yield poses

    pose_gen = pose_generator()
    for i in tqdm(range(args.n_viz)):
        render_animation(dataset.skeleton, pose_gen, cfg.t_his, ncol=12, output='./results/{}/results/'.format(args.cfg), index_i=i)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg',
                        default='h36m')
    parser.add_argument('--mode', default='train')
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--iter', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu_index', type=int, default=1)
    parser.add_argument('--n_pre', type=int, default=8)
    parser.add_argument('--n_his', type=int, default=5)
    parser.add_argument('--n_viz', type=int, default=100)
    parser.add_argument('--num_coupling_layer', type=int, default=4)
    parser.add_argument('--multimodal_threshold', type=float, default=0.5)
    args = parser.parse_args()

    """setup"""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = torch.float32
    torch.set_default_dtype(dtype)
    device = torch.device('cuda')#, index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
    # if torch.cuda.is_available():
    #     torch.cuda.set_device(args.gpu_index)
    cfg = Config(f'{args.cfg}', test=args.test)
    tb_logger = SummaryWriter(cfg.tb_dir) if args.mode == 'train' else None
    logger = create_logger(os.path.join(cfg.log_dir, 'log.txt'))

    """parameter"""
    mode = args.mode
    nz = cfg.nz
    t_his = cfg.t_his
    t_pred = cfg.t_pred
    cfg.n_his = args.n_his
    if 'n_pre' not in cfg.nf_specs.keys():
        cfg.n_pre = args.n_pre
    else:
        cfg.n_pre = cfg.nf_specs['n_pre']
    cfg.num_coupling_layer = args.num_coupling_layer
    # cfg.nz = args.nz
    """data"""
    if 'actions' in cfg.nf_specs.keys():
        act = cfg.nf_specs['actions']
    else:
        act = 'all'
    dataset_cls = DatasetH36M if cfg.dataset == 'h36m' else DatasetHumanEva
    dataset = dataset_cls('train', t_his, t_pred, actions=act, use_vel=cfg.use_vel,
                          multimodal_path=cfg.nf_specs[
                              'multimodal_path'] if 'multimodal_path' in cfg.nf_specs.keys() else None,
                          data_candi_path=cfg.nf_specs[
                              'data_candi_path'] if 'data_candi_path' in cfg.nf_specs.keys() else None)
    dataset_test = dataset_cls('test', t_his, t_pred, actions=act, use_vel=cfg.use_vel,
                               multimodal_path=cfg.nf_specs[
                                   'multimodal_path'] if 'multimodal_path' in cfg.nf_specs.keys() else None,
                               data_candi_path=cfg.nf_specs[
                                   'data_candi_path'] if 'data_candi_path' in cfg.nf_specs.keys() else None)
    if cfg.normalize_data:
        dataset.normalize_data()
        dataset_test.normalize_data(dataset.mean, dataset.std)
    traj_gt_arr = get_multimodal_gt(dataset_test)
    """model"""
    # model = get_vae_model(cfg, dataset.traj_dim)
    model, pose_prior = get_model(cfg, dataset, cfg.dataset)
    model.float()
    pose_prior.float()
    optimizer = optim.Adam(model.parameters(), lr=cfg.vae_lr)
    scheduler = get_scheduler(optimizer, policy='lambda', nepoch_fix=cfg.num_vae_epoch_fix, nepoch=cfg.num_vae_epoch)
    logger.info(">>> total params: {:.2f}M".format(
        sum(p.numel() for p in list(model.parameters())) / 1000000.0))

    cp_path = 'results/h36m_nf/models/vae_0025.p' if cfg.dataset == 'h36m' else 'results/humaneva_nf/models/vae_0025.p'
    print('loading model from checkpoint: %s' % cp_path)
    model_cp = pickle.load(open(cp_path, "rb"))
    pose_prior.load_state_dict(model_cp['model_dict'])
    pose_prior.to(device)
    # data_mean = tensor(model_cp['meta']['mean'], dtype=dtype, device=device).reshape([-1])
    # data_std = tensor(model_cp['meta']['std'], dtype=dtype, device=device).reshape([-1])

    valid_ang = pickle.load(open('./data/h36m_valid_angle.p', "rb")) if cfg.dataset == 'h36m' else pickle.load(
        open('./data/humaneva_valid_angle.p', "rb"))
    if args.iter > 0:
        cp_path = cfg.vae_model_path % args.iter
        print('loading model from checkpoint: %s' % cp_path)
        model_cp = pickle.load(open(cp_path, "rb"))
        model.load_state_dict(model_cp['model_dict'])

    if mode == 'train':
        model.to(device)
        overall_iter = 0
        stats = torch.zeros(cfg.nk)
        model.train()
        for i in range(args.iter, cfg.num_vae_epoch):
            stats = train(i, stats)
            if cfg.save_model_interval > 0 and (i + 1) % 10 == 0:
                model.eval()
                with torch.no_grad():
                    test(model, i)
                model.train()
                with to_cpu(model):
                    cp_path = cfg.vae_model_path % (i + 1)
                    model_cp = {'model_dict': model.state_dict(), 'meta': {'std': dataset.std, 'mean': dataset.mean}}
                    pickle.dump(model_cp, open(cp_path, 'wb'))

    elif mode == 'test':
        model.to(device)
        model.eval()
        with torch.no_grad():
            test(model, args.iter)

    elif mode == 'viz':
        model.to(device)
        model.eval()
        with torch.no_grad():
            visualize()
