import yaml
import os


class Config:

    def __init__(self, cfg_id, test=False, nf=False):
        self.id = cfg_id
        cfg_name = 'motion_pred/cfg/%s.yml' % cfg_id
        if not os.path.exists(cfg_name):
            print("Config file doesn't exist: %s" % cfg_name)
            exit(0)
        cfg = yaml.safe_load(open(cfg_name, 'r'))
        if nf:
            cfg_id += '_nf'
        # create dirs
        self.base_dir = 'results'

        self.cfg_dir = '%s/%s' % (self.base_dir, cfg_id)
        self.model_dir = '%s/models' % self.cfg_dir
        self.result_dir = '%s/results' % self.cfg_dir
        self.log_dir = '%s/log' % self.cfg_dir
        self.tb_dir = '%s/tb' % self.cfg_dir
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)

        # common
        self.dataset = cfg.get('dataset', 'h36m')
        self.batch_size = cfg.get('batch_size', 8)
        self.normalize_data = cfg.get('normalize_data', False)
        self.save_model_interval = cfg.get('save_model_interval', 20)
        self.t_his = cfg['t_his']
        self.t_pred = cfg['t_pred']
        self.use_vel = cfg.get('use_vel', False)

        self.nz = cfg['nz']
        self.lr = cfg['lr']
        self.dropout = cfg.get('dropout', 0.1)
        self.num_epoch = cfg['num_epoch']
        self.num_epoch_fix = cfg.get('num_epoch_fix', self.num_epoch)
        self.num_data_sample = cfg['num_data_sample']
        self.model_path = os.path.join(self.model_dir, '%04d.p')

        self.nk = cfg.get('nk', 10)
        self.nk1 = cfg.get('nk1', 5)
        self.nk2 = cfg.get('nk2', 2)
        self.lambdas = cfg.get('lambdas', [])

        self.specs = cfg.get('specs', dict())
