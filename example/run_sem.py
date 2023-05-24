import os
import argparse
import pytorch_lightning as pl

from functions import *
from core.sem.model import sem_model
from core.utils.utils import load_yaml
from core.env.cartpole_noisy import CartPoleEnv_Noise
from core.sem.synthetic import control_trajectory_multi


def main(args):
    case_path = './example'
    assert args.exp is not None, "FATAL: "+__file__+": You must specify an exp config file (e.g., *.yaml)"
    rel_path = os.path.join(case_path, '%s.yaml'%args.exp)
    cfg = load_yaml(rel_path)
    pl.seed_everything(cfg['general']['seed'])
    env = CartPoleEnv_Noise(cfg)
    X_np, GC = control_trajectory_multi(env, cfg)
    cmlp, _, _ = sem_model(X_np, cfg)
    GC_est = cmlp.GC().cpu().data.numpy()
    for i in range(cfg['general']['state_size']+2):
        GC_est[i,i] = 1
        for j in range(cfg['general']['state_size']+2):
            if i > j:
                GC_est[i,j] = 0
    gc_path = os.path.join(cfg['app_data']['dir'], "gc.npz")
    gc_est_path = os.path.join(cfg['app_data']['dir'], "gc_est.npz")
    np.savez(gc_path, GC)
    np.savez(gc_est_path, GC_est)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="proposed_algorithm")
    argparser.add_argument('-e', '--exp', type=str)
    args = argparser.parse_args()
    main(args) 