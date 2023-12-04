import torch 
import torchaudio
import torchaudio.transforms as T
from torch.nn import functional as F
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
from typing import Any, Optional

from array_setup import get_array_set_up_from_config
import numpy as np
import math

class SRPDOAcallbacks(Callback):

	def __init__(self, frame_len, frame_shift, array_type, array_setup, doa_tol, fs=16000):

		self.frame_len = frame_len
		self.frame_shift = frame_shift

		self.array_type = array_type
		self.array_setup = array_setup
		self.mic_pos = self.array_setup.mic_pos
		self.num_channels = self.mic_pos.shape[0]
		self.tolerance = doa_tol #degree

		self.freq_range = self.frame_len//2 + 1
		
		azimuths = np.deg2rad(np.linspace(0, 180, 181)) if "linear" in self.array_type else np.deg2rad(np.linspace(0, 359,360))

		theta_grid = torch.from_numpy(azimuths)
		self.u_theta = torch.stack( [torch.cos(theta_grid), torch.sin(theta_grid), torch.zeros(theta_grid.shape)], dim=1).to(torch.float32)
		pi = math.pi
		self.w = torch.from_numpy(np.array([(2*pi/self.frame_len)*k*fs for k in range(1, self.freq_range)])).unsqueeze(dim=0).to(torch.float32)

		self.ph_diff_grid = self.compute_grid_tou()
		

	def compute_grid_tou(self):
		c = 343 
		_dp_phasediff = []
		for _ch1 in range(self.num_channels):
			for _ch2 in range(_ch1+1, self.num_channels):			
				dist_mics = torch.from_numpy(self.mic_pos[_ch1] - self.mic_pos[_ch2]).unsqueeze(dim=1)
				dist_mics = torch.abs(dist_mics.to(dtype = torch.float32)) # torch.transpose(, 0, 1)
				#breakpoint()
				tou = torch.matmul(self.u_theta, dist_mics) / c
				#print(f"tou_shape: {tou.shape}, dist_mics: {dist_mics.shape}")
				out_lbl = torch.cat((torch.cos(torch.matmul(tou, self.w)), torch.sin(torch.matmul(tou, self.w))), dim=1)
				#print(f"out_lbl_shape: {out_lbl.shape}")
				_dp_phasediff.append(out_lbl.numpy()) #t()

		return torch.from_numpy(np.array(_dp_phasediff)).to(torch.float16).unsqueeze(dim=0)

	def on_test_batch_end_numpy(
		self,
		trainer: "pl.Trainer",
		pl_module: "pl.LightningModule",
		outputs: Optional[STEP_OUTPUT],
		batch: Any,
		batch_idx: int,
		dataloader_idx: int,
		) -> None:
		pass

	def on_test_batch_end(
		self,
		trainer: "pl.Trainer",
		pl_module: "pl.LightningModule",
		outputs: Optional[STEP_OUTPUT],
		batch: Any,
		batch_idx: int,
		dataloader_idx: int,
		) -> None:

		_,_, doa_azimuth_degrees = batch  #expecting doa_azimuth_degrees.shape (1, num_pairs, num_frms)
		#breakpoint()
		doa_frm = doa_azimuth_degrees[0,0,:]
		tgt_dp_phdiff_grid = self.ph_diff_grid

		est_phdiff = outputs['est_phdiff']
		est_phdiff = torch.permute(est_phdiff, (1,0,2))         #(num_micpairs, n_frames, n_freq) -> (n_frames, num_mic_pairs, 512)
		est_phdiff = est_phdiff.unsqueeze(dim=3)#.to(dtype=torch.float32)

		dot_p = torch.matmul(tgt_dp_phdiff_grid.to(device=torch.device('cuda')), est_phdiff )

		#SRP = torch.mean(dot_p.squeeze(), dim=1) / 256
		SRP = torch.mean(dot_p.squeeze(dim=-1), dim=1)

		_doA_info = torch.max(SRP, dim=1)
		_doA_est_indices  = _doA_info[1]

		#print(f"est: {_doA_est_indices}, tgt_doa_frm: {doa_frm}")
		azi_diff = torch.abs(_doA_est_indices - doa_frm)

		vad_azi_diff = torch.where(doa_frm!=-1, azi_diff, 1000)
		azimuth_result = vad_azi_diff<=self.tolerance

		vad_info = torch.where(doa_frm!=-1, 1, 0)
		#breakpoint()
		doa_acc = torch.sum(azimuth_result.float()) / torch.sum(vad_info.float())

		print(f"doa_acc: {doa_acc}")
		self.log("doa_acc", doa_acc, on_step=True,logger=True)

		return




class GradNormCallback(Callback):
	"""
	Logs the gradient norm.
	"""

	def ___on_after_backward(self, trainer: "pl.Trainer", model): 
		grad_norm_dict = gradient_norm_per_layer(model)
		self.log_dict(grad_norm_dict)

	def on_before_optimizer_step(self, trainer, model, optimizer, optimizer_idx=0): #trainer, model, 
		grad_norm_dict = gradient_norm_per_layer(model)
		self.log_dict(grad_norm_dict)

def gradient_norm_per_layer(model):
	total_norm = {}
	for layer_name, param in model.named_parameters():
		if param.grad is not None:
			param_norm = param.grad.detach().data.norm(2)
			total_norm[layer_name] = param_norm #.item() ** 2
	#total_norm = total_norm ** (1. / 2)
	return total_norm


if __name__=="__main__":


	callback = SRPDOAcallbacks(frame_len=320, frame_shift=160, array_type='linear', array_setup = get_array_set_up_from_config('linear', 8, 8))

