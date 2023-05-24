import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split


class Problem:
    def __init__(self, problem):
        self.viewer = None
        self.env = problem
        self.state_dim = problem.observation_space
        self.action_dim = problem.action_space

def Step(problem, action, itr):
    next_state, reward, done, _ = problem.env.step(action, itr)
    return next_state, reward, done, _

def RenderRandomPolicy(problem):
    action = problem.env.action_space.sample()
    return action

def GenerateTraj(problem, trial, step, cfg):
    traj = [ ]
    count = 0; add_h = 0
    max_num = cfg['init_dataset']['max_num']
    for t in range(trial):
        itr = 0
        state = problem.env.reset()
        for s in range(step):
            itr += 1
            count += 1
            action = RenderRandomPolicy(problem)
            next_state, reward, done, _ = Step(problem, action, itr)
            add_h += 1
            history = np.hstack((state, action, reward, next_state))
            state = next_state
            traj.append(history)
            if (done or count >= max_num): 
                break
        if count >= max_num:
            break
    traj = np.stack(traj, axis=1).T
    return traj, add_h

def SaveTraj(problem, traj, dir_data):
    state_dim = problem.state_dim.shape[0]
    action_dim = 1 
    st = traj[:,:state_dim]
    at = traj[:,state_dim]
    rt = traj[:,state_dim+action_dim]
    st_ = traj[:,state_dim+action_dim+1:]
    np.savez(os.path.join(dir_data, "data"), st=st, at=at, rt=rt, st_=st_)

class LoadTraj(Dataset):
	
	def __init__(self, path):
		super().__init__()
		self.path = os.path.join(path, "data.npz")
		self.traj = np.load(self.path)
		self.data = {}
		for key in ["st", "at", "rt", "st_"]:
			self.data[key] = self.traj[key]

	def __len__(self):
		return len(self.data["st"])

	def __getitem__(self, idx):
		st = torch.from_numpy(self.data["st"][idx])
		at = torch.from_numpy(np.asarray(self.data["at"][idx])).unsqueeze(0)
		rt = torch.from_numpy(np.asarray(self.data["rt"][idx])).unsqueeze(0)
		st_ = torch.from_numpy(self.data["st_"][idx])
		traj = {"st": st, "at": at, "rt": rt, "st_": st_}

		return traj

def CreateData(cfg, problem):
    dirc = cfg['app_data']['data_dir']
    step = cfg['init_dataset']['steps']
    trial = cfg['init_dataset']['trails']
    traj, add_h = GenerateTraj(problem, trial, step, cfg)
    SaveTraj(problem, traj, dirc)
    traj = LoadTraj(dirc) 
    num_validation_samples = 100
    train_data, _ = random_split(traj, [len(traj)-num_validation_samples, num_validation_samples])
    _ = DataLoader(train_data, batch_size=16, shuffle=True, pin_memory=True)
    
    return add_h

def SaveTrajIndex(problem, traj, dir_data, index):
    state_dim = problem.state_dim.shape[0]
    action_dim = 1
    st = traj[:,:state_dim]
    at = traj[:,state_dim]
    rt = traj[:,state_dim+action_dim]
    st_ = traj[:,state_dim+action_dim+1:]
    ind = index * np.ones(traj.shape[0])
    path = os.path.join(dir_data, "data.npz")
    try:
        f = open(path, 'r')
        npz = np.load(path)
        st = list(st) + list(npz["st"])
        at = list(at) + list(npz["at"])
        rt = list(rt) + list(npz["rt"])
        st_ = list(st_) + list(npz["st_"])
        ind = list(ind) + list(npz["ind"])
        np.savez(path, 
                 st = np.array(st),
                 at = np.array(at), 
                 rt = np.array(rt), 
                 st_ = np.array(st_), 
                 ind = np.array(ind))
        f.close()
    except:
        np.savez(path, st=st, at=at, rt=rt, st_=st_, ind=ind)

def CreateMultiData(cfg, problem, index):
    dirc = cfg['app_data']['data_dir']
    step = cfg['init_dataset']['steps']
    trial = cfg['init_dataset']['trails']
    traj, add_h = GenerateTraj(problem, trial, step, cfg)
    SaveTrajIndex(problem, traj, dirc, index)
    
    return add_h