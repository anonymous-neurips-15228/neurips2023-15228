import os
import gym
import torch
import collections
import numpy as np
from gym import logger
from torch import nn, optim
import pytorch_lightning as pl
from collections import OrderedDict
from torch.utils.data import DataLoader
from core.src.env.cartpole_noisy import CartPoleEnv_Noise


class DQN(nn.Module):

    def __init__(self, obs_size: int, n_actions: int, hidden_size: int = 128):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x.float())

Experience = collections.namedtuple(
    'Experience', field_names=['state', 'action', 'reward',
                               'done', 'new_state'])

class ReplayBuffer:

    def __init__(self, capacity: int) -> None:
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self) -> None:
        return len(self.buffer)

    def append(self, experience: Experience) -> None:
        self.buffer.append(experience)

    def sample(self, batch_size: int):  # -> Tuple:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])

        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),
                np.array(dones, dtype=np.bool), np.array(next_states))


class RLDataset(torch.utils.data.IterableDataset):

    def __init__(self, buffer: ReplayBuffer, sample_size: int = 200) -> None:
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self): # -> Tuple:
        states, actions, rewards, dones, new_states = self.buffer.sample(self.sample_size)
        for i in range(len(dones)):
            yield states[i], actions[i], rewards[i], dones[i], new_states[i]


class Agent:

    def __init__(self, env: gym.Env, model, replay_buffer: ReplayBuffer, test, cfg) -> None:
        self.env = env
        self.test = test
        self.model = model
        self.model_free = cfg['app_data']['model_free']
        self.model_name = cfg['app_data']['model_name']
        self.noise_size = cfg['general']['noise_size']
        self.replay_buffer = replay_buffer
        self.reset()
        self.state = self.env.reset()

    def reset(self) -> None:
        self.state = self.env.reset()

    def get_action(self, net: nn.Module, epsilon: float, device: str) -> int:
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state = torch.tensor([self.state])

            if device not in ['cpu']:
                state = state.cuda(device)

            q_values = net(state)
            _, action = torch.max(q_values, dim=1)
            action = int(action.item())

        return action

    @torch.no_grad()
    def warm_start_policy(self, cfg):
        path = os.path.join(cfg['app_data']['dir'], "data.npz")
        npz = np.load(path)
        state = npz["st"]; action = npz["at"]
        reward = npz["rt"]; new_state = npz["st_"]     
        for i in range(state.shape[0]):
            exp = Experience(state[i], action[i], reward[i], False, new_state[i])
            self.replay_buffer.append(exp)

    @torch.no_grad()
    def play_step(self, net: nn.Module, itr, epsilon: float = 0.0, device: str = 'cuda'): # -> Tuple[float, bool]:
        action = self.get_action(net, epsilon, device)
        # # do step in the environment
        if self.model_free or self.test:
            new_state, reward, done, _ = self.env.step(action, itr)
        else:
            st = torch.unsqueeze(torch.from_numpy(self.state), dim=0).to(device)
            at = torch.unsqueeze(torch.from_numpy(np.expand_dims(np.array(action),axis=0)), dim=0).to(device) 
            new_state = self.model(st, at)
            new_state = torch.squeeze(new_state).cpu().numpy()
            
            done = bool(
                new_state[0] < -self.env.x_threshold
                or new_state[0] > self.env.x_threshold
                or new_state[2] < -self.env.theta_threshold_radians
                or new_state[2] > self.env.theta_threshold_radians
            )

            if not done:
                reward = 1.0
            elif self.env.steps_beyond_done is None:
                # Pole just fell!
                self.env.steps_beyond_done = 0
                reward = 1.0
            else:
                if self.env.steps_beyond_done == 0:
                    logger.warn()
                self.env.steps_beyond_done += 1
                reward = 0.0

        exp = Experience(self.state, action, reward, done, new_state)
        self.replay_buffer.append(exp)

        self.state = new_state
        if done:
            self.reset()
        return reward, done


class DQNLightning(pl.LightningModule):

    def __init__(self, cfg, scm_model, test) -> None:
        super().__init__()
        self.save_hyperparameters("cfg")
        self.env = CartPoleEnv_Noise(cfg)
        self.model_free = cfg['app_data']['model_free']
        self.replay_size = cfg['dqn_paras']['replay_size']
        self.warm_start_steps = cfg['dqn_paras']['warm_start_steps']
        self.batch_size = cfg['dqn_paras']['batch_size']
        self.episode_length = cfg['dqn_paras']['episode_length']
        self.gamma = cfg['dqn_paras']['gamma']
        self.eps_end = cfg['dqn_paras']['eps_end']
        self.eps_start = cfg['dqn_paras']['eps_start']
        self.eps_last_frame = cfg['dqn_paras']['eps_last_frame']
        self.sync_rate = cfg['dqn_paras']['sync_rate']
        self.lr_dqn = cfg['dqn_paras']['lr_dqn']

        if self.model_free or test:
            self.model = self.env
        else:
            self.model = scm_model.forward
        obs_size = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.n

        self.net = DQN(obs_size, n_actions)
        self.target_net = DQN(obs_size, n_actions)

        self.buffer = ReplayBuffer(self.replay_size)
        self.agent = Agent(self.env, self.model, self.buffer, test, cfg)

        self.index = 0
        self.total_reward = 0
        self.episode_reward = 0
        self.warm_start(cfg)
        self.populate(self.warm_start_steps)
        self.steps = cfg['dqn_paras']['end_trails']

    def warm_start(self, cfg):
       self.agent.warm_start_policy(cfg)

    def populate(self, steps: int = 1000) -> None:
        for i in range(steps):
            self.agent.play_step(self.net, i, epsilon=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.net(x)
        return output

    def dqn_mse_loss(self, batch) -> torch.Tensor: 
        states, actions, rewards, dones, next_states = batch

        state_action_values = self.net(states).gather(1, actions.long().unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_state_values = self.target_net(next_states).max(1)[0]
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * self.gamma + rewards

        return nn.MSELoss()(state_action_values, expected_state_action_values)

    def training_step(self, batch, nb_batch) -> OrderedDict: 
        self.index += 1
        device = self.get_device(batch)
        epsilon = max(self.eps_end, self.eps_start -
                      self.global_step + 1 / self.eps_last_frame)

        reward, done = self.agent.play_step(self.net, self.index, epsilon, device)
        self.episode_reward += reward
        loss = self.dqn_mse_loss(batch).unsqueeze(0)

        if done or self.index == self.steps:
            self.total_reward = self.episode_reward
            self.log('training_index', self.index)
            self.index = 0
            self.episode_reward = 0
            self.log('training_dqn', torch.tensor(self.total_reward).to(device))

        if self.global_step % self.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        log = {'total_reward': torch.tensor(self.total_reward).to(device),
               'reward': torch.tensor(reward).to(device),
               'steps': torch.tensor(self.global_step).to(device)}
        status = {
            'steps': torch.tensor(self.global_step).to(device),
            'total_reward': torch.tensor(self.total_reward).to(device)
        }

        return OrderedDict({'loss': loss, 'log': log, 'progress_bar': status})

    def configure_optimizers(self):  
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr_dqn)
        return [optimizer]

    def train_dataloader(self) -> DataLoader:
        dataset = RLDataset(self.buffer, self.episode_length)
        dataloader = DataLoader(dataset=dataset,
                                batch_size=self.batch_size,
                                )
        return dataloader

    def get_device(self, batch) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"
