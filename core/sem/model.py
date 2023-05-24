import torch
import numpy as np
from datetime import datetime
from .models.cmlp import cMLP, train_model_ista

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sem_model(X_np, cfg):
    X = torch.tensor(X_np[np.newaxis], dtype=torch.float32, device=device) 
    cmlp = cMLP(X.shape[-1], lag=1, hidden=[cfg['scm_paras']['hidden_size']]).cuda(device=device)
    train_loss_list = train_model_ista(cmlp, X, 
                                       lam=0.002, lam_ridge=1e-2, 
                                       lr=5e-2, penalty='GSGL', 
                                       max_iter=int(cfg['scm_paras']['epoch']), check_every=1000)

    return cmlp, train_loss_list, _
