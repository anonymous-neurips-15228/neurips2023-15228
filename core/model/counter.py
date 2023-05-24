import sys
import torch
import numpy as np
import torch.nn as nn
sys.path.append("..")
import pytorch_lightning as pl

from .mlp import NLayerLeakyMLP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.state_size = cfg['general']['state_size']
        self.noise_size = cfg['general']['noise_size']
        self.hidden_size = cfg['general']['hidden_size']
        self.layer = nn.Sequential(nn.Linear(self.state_size, self.hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(self.hidden_size, self.hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(self.hidden_size, self.hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(self.hidden_size, self.noise_size))

    def forward(self, x):
        out = self.layer(x)
        return out


class Generator(nn.Module):

    def __init__(self, GC_est, cfg):
        super().__init__()
        out_features = 1 
        self.layer_list = [ ]
        self.index_list = [ ]
        self.type = cfg['general']['type']
        self.state_dim = cfg['general']['state_size']
        hidden_dim = cfg['general']['hidden_size']
        num_layers = cfg['gan_paras']['layer_num']

        for i in range(self.state_dim):
            sem_i = GC_est[i, :]
            index_i = torch.squeeze(torch.tensor(np.where(sem_i==1))).to(device)
            self.index_list.append(index_i)
            in_features = np.sum(sem_i) + 1 
            layer = NLayerLeakyMLP(in_features, out_features, num_layers, hidden_dim)
            self.layer_list.append(layer)
        self.layer_list = nn.ModuleList(self.layer_list)

    def forward(self, y, z, index, normal=False):
        ns = [ ]
        for i in range(self.state_dim):
            z_i = torch.unsqueeze(z[:, i], dim=-1)
            if self.type == "multi":
                select_idx = self.index_list[i]
                if index == "target":
                    index = 3 * torch.ones(z_i.shape).to(device)
                if (select_idx==self.state_dim+1).cpu().numpy().any():
                    select_idx = select_idx[:-1]
                    paraent_i = y.index_select(1, select_idx)
                    out_i = torch.cat([paraent_i, z_i, index.float()], dim=-1)
                else:
                    paraent_i = y.index_select(1, select_idx)
                    out_i = torch.cat([paraent_i, z_i], dim=-1)                    
            elif self.type == "single":
                paraent_i = y.index_select(1, self.index_list[i])
                out_i = torch.cat([paraent_i, z_i], dim=-1)   
            output = self.layer_list[i](out_i)
            ns.append(output)
        embedding = torch.cat((ns[0], ns[1], ns[2], ns[3], ns[4], ns[5]), dim=-1) 
        return embedding


class Discriminator(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.state_size = cfg['general']['state_size']
        self.noise_size = cfg['general']['noise_size']
        self.hidden_size = cfg['general']['hidden_size']
        self.con_size = cfg['general']['condition_size']
        self.embedding = nn.Linear(self.con_size, self.con_size)
        self.layer = nn.Sequential(nn.Linear(self.state_size+self.con_size+self.noise_size, self.hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(self.hidden_size, self.hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(self.hidden_size, self.hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(self.hidden_size, 1),
                                   nn.Sigmoid())

    def forward(self, x, y, z):
        out = torch.cat([x, z, y], dim=-1)
        out = self.layer(out)
        return out


class LitCounter(pl.LightningModule):

    def __init__(self, GC_est, cfg):
        super().__init__()
        self.save_hyperparameters("cfg")
        self.lr_gan = cfg['gan_paras']['lr_gan']
        self.noise_size = cfg['general']['noise_size']
        self.encoder = Encoder(cfg)
        self.generator = Generator(GC_est, cfg)
        self.discriminator = Discriminator(cfg)
        self.dis_cogan = cfg['gan_paras']['dis_cogan']
        self.loss_fn = torch.nn.MSELoss()

    def dataNormal(self, x):
        d_min = x.min()
        d_max = x.max()
        dst = d_max - d_min
        d_norm = (x - d_min).true_divide(dst)
        return d_norm
    
    def forward(self, st, at):
        z = torch.randn(st.shape[0], self.noise_size, device=device).float()
        y = torch.cat([st, at], dim=-1).float()
        index = "target"
        return self.generator(y, z, index)
    
    def counterfactual(self, st, counter_at, st_):
        z = self.encoder(torch.from_numpy(st_).float().to(device))
        counter_at = torch.unsqueeze(torch.from_numpy(np.expand_dims(np.array(counter_at),axis=0)), dim=0).to(device)
        y = torch.cat([st.to(device), counter_at], dim=-1).float()
        index = "target"
        next_state = self.generator(y, torch.unsqueeze(z, dim=0), index, normal=False)
        return next_state
        
    def generator_step(self, x, y, index):
        y = y.to(device).float()
        z = torch.randn(x.shape[0], self.noise_size, device=device).float()
        generated_states = self.generator(y, z, index)
        dg_output = torch.squeeze(self.discriminator(generated_states, y, z))
        return dg_output, generated_states

    def discriminator_step(self, x, y):
        encoded_noise = self.encoder(x.float())
        de_output = torch.squeeze(self.discriminator(x.float(), y.float(), encoded_noise))
        return de_output

    def training_step(self, batch, batch_idx, optimizer_idx):
        eps = 1e-10
        st = batch['st']; at = batch['at']
        rt = batch['rt']; st_ = batch['st_']
        x = st_; y = torch.cat([st, at], dim=-1)
        index = batch['ind']
        DG, generated_states = self.generator_step(x, y, index)
        DE = self.discriminator_step(x, y)
        error_g = self.loss_fn(generated_states.float(), x.float())

        error = torch.abs(x - generated_states)
        d_norm = (1 - torch.mean(torch.mean(self.dataNormal(error), dim=0), dim=0)).float()

        loss_D = torch.log(DE + eps) + torch.log(1 - d_norm*DG + eps)
        loss_EG = torch.log(d_norm*DG + eps) + torch.log(1 - DE + eps)

        if optimizer_idx == 0:
            if self.dis_cogan:
                loss = -0.01 * torch.mean(loss_EG) + error_g 
            else:
                loss = -torch.mean(loss_EG) 
            loss.requires_grad_()

        if optimizer_idx == 1:
            loss = -torch.mean(loss_D)
            loss.requires_grad_()

        if optimizer_idx == 0:
            self.log('loss_gen', -torch.mean(loss_EG))
            self.log('loss_error', error_g)
        else:
            self.log('loss_dis', loss)
            
        return loss

    def configure_optimizers(self):
        eg_optimizer = torch.optim.Adam(list(self.encoder.parameters())+list(self.generator.parameters()), lr=self.lr_gan, betas=(0.5, 0.999), weight_decay=1e-5)
        d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr_gan, betas=(0.5, 0.999), weight_decay=1e-5)
        return [eg_optimizer, d_optimizer], []
