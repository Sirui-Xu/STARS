import os
import sys
import pickle
import argparse
import time
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())
from utils import *
from motion_pred.utils.config import Config
from motion_pred.utils.dataset_h36m import DatasetH36M
from motion_pred.utils.dataset_humaneva import DatasetHumanEva
from models.motion_pred import *
from utils import util


def loss_function(prior_lkh, log_det_jacobian):
    loss_p = -prior_lkh.mean()
    loss_jac = - log_det_jacobian.mean()
    loss_r = loss_p + loss_jac

    return loss_r, np.array([loss_r.item(), loss_p.item(), loss_jac.item()])


def train(epoch):
    model.train()
    t_s = time.time()
    train_losses = 0
    train_grad = 0
    total_num_sample = 0
    loss_names = ['LKH', 'log_p(z)', 'log_det']
    generator = dataset.sampling_generator(num_samples=cfg.num_vae_data_sample, batch_size=cfg.batch_size)
    prior = torch.distributions.Normal(torch.tensor(0, dtype=dtype, device=device),
                                       torch.tensor(1, dtype=dtype, device=device))
    for traj_np in generator:
        with torch.no_grad():
            traj_np = traj_np[:, 0]
            traj_np[:, 0] = 0
            traj_np = util.absolute2relative(traj_np, parents=dataset.skeleton.parents())
            traj = tensor(traj_np, device=device, dtype=dtype)  # .permute(0, 2, 1).contiguous()
            bs, nj, _ = traj.shape
            x = traj.reshape([bs, -1])

        z, log_det_jacobian = model(x)
        prior_likelihood = prior.log_prob(z).sum(dim=1)

        loss, losses = loss_function(prior_likelihood, log_det_jacobian)
        optimizer.zero_grad()
        loss.backward()
        # grad_norm = torch.nn.utils.clip_grad_norm_(list(model.parameters()), max_norm=10000)
        grad_norm = 0
        train_grad += grad_norm
        optimizer.step()
        train_losses += losses
        total_num_sample += 1
        del loss, z, prior_likelihood, log_det_jacobian

    scheduler.step()
    # dt = time.time() - t_s
    train_losses /= total_num_sample
    lr = optimizer.param_groups[0]['lr']
    losses_str = ' '.join(['{}: {:.4f}'.format(x, y) for x, y in zip(loss_names, train_losses)])

    # average cost of log time 20s
    tb_logger.add_scalar('train_grad', train_grad / total_num_sample, epoch)
    for name, loss in zip(loss_names, train_losses):
        tb_logger.add_scalars('vae_' + name, {'train': loss}, epoch)

    logger.info('====> Epoch: {} Time: {:.2f} {} lr: {:.5f}'.format(epoch, time.time() - t_s, losses_str, lr))


def val(epoch):
    model.eval()
    t_s = time.time()
    train_losses = 0
    total_num_sample = 0
    loss_names = ['LKH', 'log_p(z)', 'log_det']
    generator = dataset.sampling_generator(num_samples=cfg.num_vae_data_sample // 2, batch_size=cfg.batch_size)
    prior = torch.distributions.Normal(torch.tensor(0, dtype=dtype, device=device),
                                       torch.tensor(1, dtype=dtype, device=device))
    loginfos = None
    with torch.no_grad():
        xx = []
        for traj_np in generator:
            traj_np = traj_np[:, 0]
            traj_np[:, 0] = 0
            traj_np = util.absolute2relative(traj_np, parents=dataset.skeleton.parents())
            traj = tensor(traj_np, device=device, dtype=dtype)  # .permute(0, 2, 1).contiguous()
            bs, nj, _ = traj.shape
            x = traj.reshape([bs, -1])

            z, log_det_jacobian = model(x)
            prior_likelihood = prior.log_prob(z).sum(dim=1)
            loginf = {}
            loginf[f'z'] = z.cpu().data.numpy()

            loss, losses = loss_function(prior_likelihood, log_det_jacobian)

            train_losses += losses
            total_num_sample += 1
            del loss, z, prior_likelihood, log_det_jacobian
            loginfos = combine_dict(loginf, loginfos)

    # dt = time.time() - t_s
    train_losses /= total_num_sample
    lr = optimizer.param_groups[0]['lr']
    losses_str = ' '.join(['{}: {:.4f}'.format(x, y) for x, y in zip(loss_names, train_losses)])

    # average cost of log time 20s
    for name, loss in zip(loss_names, train_losses):
        tb_logger.add_scalars('vae_' + name, {'val': loss}, epoch)
    logger.info('====> Epoch: {} Val Time: {:.2f} {} lr: {:.5f}'.format(epoch, time.time() - t_s, losses_str, lr))

    t_s = time.time()
    generator = dataset_test.sampling_generator(num_samples=cfg.num_vae_data_sample // 2, batch_size=cfg.batch_size)
    loginfos_test = None
    with torch.no_grad():
        xx = []
        for traj_np in generator:
            traj_np = traj_np[:, 0]
            traj_np[:, 0] = 0
            traj_np = util.absolute2relative(traj_np, parents=dataset.skeleton.parents())
            traj = tensor(traj_np, device=device, dtype=dtype)  # .permute(0, 2, 1).contiguous()
            bs, nj, _ = traj.shape
            x = traj.reshape([bs, -1])

            z, log_det_jacobian = model(x)
            prior_likelihood = prior.log_prob(z).sum(dim=1)
            loginf = {}
            loginf[f'z'] = z.cpu().data.numpy()

            loss, losses = loss_function(prior_likelihood, log_det_jacobian)

            train_losses += losses
            total_num_sample += 1
            del loss, z, prior_likelihood, log_det_jacobian
            loginfos_test = combine_dict(loginf, loginfos_test)

    # dt = time.time() - t_s
    train_losses /= total_num_sample
    lr = optimizer.param_groups[0]['lr']
    losses_str = ' '.join(['{}: {:.4f}'.format(x, y) for x, y in zip(loss_names, train_losses)])

    # average cost of log time 20s
    for name, loss in zip(loss_names, train_losses):
        tb_logger.add_scalars('vae_' + name, {'test': loss}, epoch)
    logger.info('====> Epoch: {} Test Time: {:.2f} {} lr: {:.5f}'.format(epoch, time.time() - t_s, losses_str, lr))

    t_s = time.time()
    zz = loginfos['z']
    zz = zz.reshape([zz.shape[0], -1])
    bs, data_dim = zz.shape
    zz_test = loginfos_test['z']
    zz_test = zz_test.reshape([zz.shape[0], -1])
    # bs, data_dim = zz_test.shape
    for ii in range(data_dim):
        fig = plt.figure()
        ax1 = plt.subplot(121)
        _ = plt.hist(zz[:, ii].reshape(-1), bins=100, density=True, alpha=0.5, color='b')
        x = torch.from_numpy(np.arange(-5, 5, 0.01)).float().to(device)
        y = prior.cdf(x)
        x = x[1:]
        y = (y[1:] - y[:-1]) * 100
        plt.plot(x.cpu().data, y.cpu().data)
        ax1.set_title(f'z_val_{ii}')
        ax2 = plt.subplot(122)
        _ = plt.hist(zz_test[:, ii].reshape(-1), bins=100, density=True, alpha=0.5, color='b')
        x = torch.from_numpy(np.arange(-5, 5, 0.01)).float().to(device)
        y = prior.cdf(x)
        x = x[1:]
        y = (y[1:] - y[:-1]) * 100
        plt.plot(x.cpu().data, y.cpu().data)
        ax2.set_title(f'z_test_{ii}')
        tb_logger.add_figure(f'z_{ii}', fig, epoch)
        plt.clf()
        plt.cla()
        plt.close(fig)

    # plot covariance matrix
    fig = plt.figure()
    ax1 = plt.subplot(121)
    # zz = zz.reshape([bs, -1])
    zz = zz - zz.mean(axis=0)
    cov = 1 / bs * np.abs(np.matmul(zz.transpose([1, 0]), zz))
    std = np.sqrt(np.diag(cov))[:, None] * np.sqrt(np.diag(cov))[None, :]
    corr = cov / (std + 1e-10) - np.eye(cov.shape[0])
    plt.imshow(corr)
    plt.colorbar()
    ax1.set_title('z_val_corr')

    ax2 = plt.subplot(122)
    # zz = zz.reshape([bs, -1])
    zz_test = zz_test - zz_test.mean(axis=0)
    cov = 1 / bs * np.abs(np.matmul(zz_test.transpose([1, 0]), zz_test))
    std = np.sqrt(np.diag(cov))[:, None] * np.sqrt(np.diag(cov))[None, :]
    corr = cov / (std + 1e-10) - np.eye(cov.shape[0])
    plt.imshow(corr)
    plt.colorbar()
    ax2.set_title('z_test_corr')
    tb_logger.add_figure('z_corr', plt.gcf(), epoch)
    plt.close(fig)
    print(f'>>>>log time {time.time() - t_s:.3f}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='h36m_nf')
    parser.add_argument('--mode', default='train')
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--iter', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu_index', type=int, default=1)
    parser.add_argument('--n_pre', type=int, default=10)
    parser.add_argument('--n_his', type=int, default=5)
    parser.add_argument('--trial', type=int, default=1)
    parser.add_argument('--num_coupling_layer', type=int, default=6)
    parser.add_argument('--nz', type=int, default=10)
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
    cfg.n_pre = args.n_pre
    cfg.num_coupling_layer = args.num_coupling_layer
    cfg.nz = args.nz
    """data"""
    dataset_cls = DatasetH36M if cfg.dataset == 'h36m' else DatasetHumanEva
    dataset = dataset_cls('train', t_his, t_pred, actions='all', use_vel=cfg.use_vel)
    dataset_test = dataset_cls('test', t_his, t_pred, actions='all', use_vel=cfg.use_vel)
    if cfg.normalize_data:
        dataset.normalize_data()
        dataset_test.normalize_data(dataset.mean, dataset.std)

    """model"""
    model = get_model(cfg, dataset.traj_dim, args.cfg)
    print(model)
    model.float()
    optimizer = optim.Adam(model.parameters(), lr=cfg.vae_lr, weight_decay=1e-3)
    scheduler = get_scheduler(optimizer, policy='lambda', nepoch_fix=cfg.num_vae_epoch_fix, nepoch=cfg.num_vae_epoch)
    logger.info(">>> total params: {:.5f}M".format(sum(p.numel() for p in list(model.parameters())) / 1000000.0))

    if args.iter > 0:
        cp_path = cfg.vae_model_path % args.iter
        print('loading model from checkpoint: %s' % cp_path)
        model_cp = pickle.load(open(cp_path, "rb"))
        model.load_state_dict(model_cp['model_dict'])

    if mode == 'train':
        model.to(device)
        overall_iter = 0
        for i in range(args.iter, cfg.num_vae_epoch):
            model.train()
            train(i)
            model.eval()
            val(i)
            # test(i)
            if cfg.save_model_interval > 0 and (i + 1) % cfg.save_model_interval == 0:
                with to_cpu(model):
                    cp_path = cfg.vae_model_path % (i + 1)
                    model_cp = {'model_dict': model.state_dict(), 'meta': {'std': dataset.std, 'mean': dataset.mean}}
                    pickle.dump(model_cp, open(cp_path, 'wb'))
