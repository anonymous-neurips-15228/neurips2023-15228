import os
import sys
import argparse
sys.path.append('../../')

from core.src.utils.utils import load_yaml, setup_seed
from core.src.utils.init_dataset import Problem, CreateMultiData
from core.src.env.cartpole_noisy_multi import CartPoleEnv_Noise_Multi

def random_multi_data(env, cfg, index):
    problem = Problem(env)
    add_h = CreateMultiData(cfg, problem=problem, index=index)
    return add_h

def main(args):
    case_path = './example'
    assert args.exp is not None, "FATAL: "+__file__+": You must specify an exp config file (e.g., *.yaml)"
    rel_path = os.path.join(case_path, '%s.yaml'%args.exp)
    cfg = load_yaml(rel_path)

    add_h = 0
    env_index = [1, 2, 4, 5]
    para_index = [cfg['trans_paras']['para1'], 
                  cfg['trans_paras']['para2'], 
                  cfg['trans_paras']['para4'], 
                  cfg['trans_paras']['para5']]
    for i in range(len(env_index)):
        env = CartPoleEnv_Noise_Multi(cfg, para_index[i])
        addh = random_multi_data(env, cfg, env_index[i]) 
        add_h += addh

if __name__ == "__main__":
    seed = 124
    setup_seed(seed)
    argparser = argparse.ArgumentParser(description="proposed_algorithm")
    argparser.add_argument('-e', '--exp', type=str)
    args = argparser.parse_args()
    main(args) 