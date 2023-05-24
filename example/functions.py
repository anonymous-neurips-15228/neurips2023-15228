import os
import torch
import numpy as np
from gym import logger
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from core.policy.dqn import DQNLightning
from core.model.counter import LitCounter
from core.env.cartpole_noisy import CartPoleEnv_Noise

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def SCM(data, GC_est, cfg, case_path):
    num_validation_samples = 1000
    ckp_path = os.path.join(case_path, "./result/checkpoints/"+"/"
                            +cfg['general']['env_name']+"/"
                            +cfg['app_data']['model_name'])
    train_data, val_data = random_split(data, [len(data)-num_validation_samples, num_validation_samples])
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

    model = LitCounter(GC_est, cfg).to(device)
    trainer = pl.Trainer(default_root_dir=ckp_path, 
                         gpus=1 if torch.cuda.is_available() else None,
                         max_epochs=cfg['general']['epoch_scm'], 
                         progress_bar_refresh_rate=0)         
    trainer.fit(model, train_loader, val_loader)
    save_path = os.path.join(case_path, "./result/checkpoints/"+"/"
                            +cfg['general']['env_name']+"/"
                            +cfg['app_data']['model_name']+"/scm.ckpt")
    trainer.save_checkpoint(save_path)

    return model

def DQN(scm_model, cfg, case_path):
    ckp_path = os.path.join(case_path, "./result/checkpoints/"+"/"
                            +cfg['general']['env_name']+"/"
                            +cfg['app_data']['model_name']+"/"
                            +cfg['general']['policy_name'])

    save_path = os.path.join(case_path, "./result/checkpoints/"+"/"
                            +cfg['general']['env_name']+"/"
                            +cfg['app_data']['model_name']+"/"
                            +cfg['general']['policy_name']+"/dqn.ckpt")

    try: 
        policy = DQNLightning.load_from_checkpoint(checkpoint_path=save_path, 
                                                    scm_model=scm_model.to(device), 
                                                    test=True).to(device)            
    except:
        policy = DQNLightning(cfg, scm_model=scm_model.to(device), test=False).to(device)

    trainer = pl.Trainer(default_root_dir=ckp_path, 
                         gpus=1 if torch.cuda.is_available() else None,
                         max_epochs=cfg['general']['epoch_dqn'],
                         progress_bar_refresh_rate=0,
                         val_check_interval=100)

    trainer.fit(policy)

    save_path = os.path.join(case_path, "./result/checkpoints/"+"/"
                            +cfg['general']['env_name']+"/"
                            +cfg['app_data']['model_name']+"/"
                            +cfg['general']['policy_name']+"/dqn.ckpt")
    trainer.save_checkpoint(save_path)

    return policy

def counterfactual(scm_model, cfg):
    traj = []
    scm_model = scm_model.to(device)
    env = CartPoleEnv_Noise(cfg)
    path = os.path.join(cfg['app_data']['dir'], "data.npz")
    npz = np.load(path)
    state_dim = cfg['general']['state_size']
    action_dim = cfg['general']['action_size']
    H_st = npz["st"]; H_at = np.expand_dims(npz["at"], axis=1)
    H_rt = np.expand_dims(npz["rt"], axis=1); H_st_ = npz["st_"]
    H = np.hstack((H_st, H_at, H_rt, H_st_))
    np.random.shuffle(H)
    for i in range(int(H.shape[0])):
        h = H[i]
        state = h[:state_dim]; action = h[state_dim]
        next_state = h[state_dim+action_dim+1:]
        st = torch.unsqueeze(torch.from_numpy(state), dim=0)
        at = int(action)
        counter_at = ~at+2
        new_state = scm_model.counterfactual(st, counter_at, next_state)
        new_state = torch.squeeze(new_state).detach().cpu().numpy()
        done = bool(
            new_state[0] < -env.x_threshold
            or new_state[0] > env.x_threshold
            or new_state[2] < -env.theta_threshold_radians
            or new_state[2] > env.theta_threshold_radians
        )

        if not done:
            reward = 1.0
        elif env.steps_beyond_done is None:
            env.steps_beyond_done = 0
            reward = 1.0
        else:
            if env.steps_beyond_done == 0:
                logger.warn()
            env.steps_beyond_done += 1
            reward = 0.0
        history = np.hstack((state, counter_at, reward, new_state))
        traj.append(history)

    traj = np.stack(traj, axis=1).T
    ind = 3 * np.ones(traj.shape[0])
    st = list(traj[:,:state_dim]) + list(npz["st"])
    at = list(traj[:,state_dim]) + list(npz["at"])
    rt = list(traj[:,state_dim+action_dim]) + list(npz["rt"])
    st_ = list(traj[:,state_dim+action_dim+1:]) + list(npz["st_"])
    ind = list(ind) + list(npz["ind"])
    np.savez(path, 
             st=np.array(st), 
             at=np.array(at), 
             rt=np.array(rt), 
             st_=np.array(st_), 
             ind = np.array(ind))

def test(scm_model, policy, case_path, cfg):
    ckp_path = os.path.join(case_path, "./result/checkpoints/"+"/"
                            +cfg['general']['env_name']+"/"
                            +cfg['app_data']['model_name']+"/"
                            +cfg['general']['policy_name'])
    save_path = os.path.join(case_path, "./result/checkpoints/"+"/"
                            +cfg['general']['env_name']+"/"
                            +cfg['app_data']['model_name']+"/"
                            +cfg['general']['policy_name']+"/dqn.ckpt")

    policy = DQNLightning.load_from_checkpoint(checkpoint_path=save_path, 
                                                   scm_model=scm_model.to(device), 
                                                   test=True).to(device)
    trainer = pl.Trainer(default_root_dir=ckp_path, 
                         gpus=1 if torch.cuda.is_available() else None,
                         max_epochs=cfg['general']['epoch_dqn'],
                         progress_bar_refresh_rate=0,
                         val_check_interval=100)

    trainer.fit(policy)

def Exexution(scm_model, policy, cfg):
    add_h = 0
    traj = [ ]
    env = CartPoleEnv_Noise(cfg)
    scm_model = scm_model.to(device)

    for episode in range(cfg['general']['episodes']):
        done = False
        state = env.reset()
        for j in range(cfg['general']['steps']):
            add_h += 1
            q_values = policy.net(torch.tensor([state]))
            _, action = torch.max(q_values, dim=1)
            action = int(action.item())
            st = torch.unsqueeze(torch.from_numpy(state), dim=0).to(device)
            at = torch.unsqueeze(torch.from_numpy(np.expand_dims(np.array(action),axis=0)), dim=0).to(device)
            new_state = scm_model.forward(st, at)
            next_state = torch.squeeze(new_state).detach().cpu().numpy()

            done = bool(
                next_state[0] < -env.x_threshold
                or next_state[0] > env.x_threshold
                or next_state[2] < -env.theta_threshold_radians
                or next_state[2] > env.theta_threshold_radians
            )

            if not done:
                reward = 1.0
            elif env.steps_beyond_done is None:
                # Pole just fell!
                env.steps_beyond_done = 0
                reward = 1.0
            else:
                if env.steps_beyond_done == 0:
                    logger.warn()
                env.steps_beyond_done += 1
                reward = 0.0

            history = np.hstack((state, action, reward, next_state))
            state = next_state
            traj.append(history)

            if done:
                state = env.reset()
                break
    
    path = os.path.join(cfg['app_data']['dir'], "data.npz")
    npz = np.load(path)
    traj = np.stack(traj, axis=1).T
    ind = 3 * np.ones(traj.shape[0])
    state_dim = cfg['general']['state_size']
    action_dim = cfg['general']['action_size']
    st = list(traj[:,:state_dim]) + list(npz["st"])
    at = list(traj[:,state_dim]) + list(npz["at"])
    rt = list(traj[:,state_dim+action_dim]) + list(npz["rt"])
    st_ = list(traj[:,state_dim+action_dim+1:]) + list(npz["st_"])
    ind = list(ind) + list(npz["ind"])
    np.savez(path, 
             st=np.array(st), 
             at=np.array(at), 
             rt=np.array(rt), 
             st_=np.array(st_), 
             ind = np.array(ind))
    return add_h
