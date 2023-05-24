import os
import sys
import time
import argparse
sys.path.append('../../')
from datetime import datetime
import pytorch_lightning as pl

from functions import *
from core.utils.utils import load_yaml
from core.env.cartpole_noisy import CartPoleEnv_Noise
from core.utils.dataloader import SimulationMultiDataset
from core.sem.synthetic import control_trajectory_multi


def main(args):
    case_path = './example'
    assert args.exp is not None, "FATAL: "+__file__+": You must specify an exp config file (e.g., *.yaml)"
    rel_path = os.path.join(case_path, '%s.yaml'%args.exp)
    cfg = load_yaml(rel_path)
    pl.seed_everything(cfg['general']['seed'])
    env = CartPoleEnv_Noise(cfg)
    _, GC_est = control_trajectory_multi(env, cfg)
    data = SimulationMultiDataset(dirc=cfg['app_data']['dir'])
    scm_model = SCM(data, GC_est, cfg, case_path)
    scm_model.eval()
    print(scm_model.eval())
    policy = DQN(scm_model, data, cfg, case_path)
    policy.eval()
    _ = Exexution(scm_model, policy, cfg, case_path)
    counterfactual(scm_model, cfg)
    data = SimulationMultiDataset(dirc=cfg['app_data']['dir'])
    policy = DQN(scm_model, data, cfg, case_path)
    policy.eval()
    test(scm_model, policy, case_path, cfg)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="proposed_algorithm")
    argparser.add_argument('-e', '--exp', type=str)
    args = argparser.parse_args()
    main(args) 