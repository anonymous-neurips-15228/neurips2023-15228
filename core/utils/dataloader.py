import os
import torch
import numpy as np
from torch.utils.data import Dataset


class SimulationMultiDataset(Dataset):
	
	def __init__(self, dirc):
		super().__init__()
		self.path = os.path.join(dirc, "data.npz")
		self.path = self.path.replace('\\','/')
		self.npz = np.load(self.path)
		self.data = {}
		for key in ["st", "at", "rt", "st_", "ind"]:
			self.data[key] = self.npz[key]

	def __len__(self):
		return len(self.data["st"])

	def __getitem__(self, idx):
		st = torch.from_numpy(self.data["st"][idx])
		at = torch.from_numpy(np.asarray(self.data["at"][idx])).unsqueeze(0)
		rt = torch.from_numpy(np.asarray(self.data["rt"][idx])).unsqueeze(0)
		st_ = torch.from_numpy(self.data["st_"][idx])
		index = torch.from_numpy(np.asarray(self.data["ind"][idx])).unsqueeze(0)
		sample = {"st": st, "at": at, "rt": rt, "st_": st_, "ind": index}

		return sample