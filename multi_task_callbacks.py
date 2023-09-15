import torch 
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from torch.nn import functional as F
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT

from typing import Any, Optional

import numpy as np
import os 
import csv
from metrics import eval_metrics_batch_v1

class GradNormCallback(Callback):
	"""
	Logs the gradient norm.
	"""

	def ___on_after_backward(self, trainer: "pl.Trainer", model): 
		grad_norm_dict = gradient_norm_per_layer(model)
		self.log_dict(grad_norm_dict)

	def on_before_optimizer_step(self, trainer, model, optimizer, optimizer_idx=0): #trainer, model, 
		grad_norm_dict, grad_data_ratio = gradient_norm_per_layer(model)
		#print(grad_norm_dict, grad_data_ratio)
		self.log_dict(grad_norm_dict)
		self.log_dict(grad_data_ratio)

def gradient_norm_per_layer(model):
	total_norm = {}
	grad_data_ratio = {}
	for layer_name, param in model.named_parameters():
		#breakpoint()
		if param.grad is not None:
			#print(param.grad)
			param_grad_norm = param.grad.detach().data.norm(2)
			param_norm = param.data.norm(2)
			total_norm[layer_name] = param_grad_norm #.item() ** 2


			param_std = torch.std(param.data)
			param_grad_std = torch.std(param.grad.detach().data)
			grad_data_ratio[layer_name] = param_grad_std/param_std
	#total_norm = total_norm ** (1. / 2)
	return total_norm, grad_data_ratio


class multi_task_callback(Callback):
	def __init__(self, doa_tol, se_metrics_flag):
		self.doa_tol = doa_tol
		self.frame_size = 320
		self.frame_shift = 160
		self.se_metrics_flag = se_metrics_flag
		
	def format_istft(self, mix_ri_spec):
		#adjusting shape, type for istft
		#(batch_size, num_ch, T, F) . -> (batch_size*(num_ch/2), F, T, 2)
		mix_ri_spec = mix_ri_spec.to(torch.float32)

		(batch_size, n_ch, n_frms, n_freq) = mix_ri_spec.shape

		intm_mix_ri_spec = torch.reshape(mix_ri_spec, (batch_size*n_ch, n_frms, n_freq))
		intm_mix_ri_spec = torch.reshape(intm_mix_ri_spec, (intm_mix_ri_spec.shape[0]//2, 2, n_frms, n_freq))
		_mix_ri_spec = torch.permute(intm_mix_ri_spec,[0,3,2,1])

		mix_sig = torch.istft(_mix_ri_spec, self.frame_size, self.frame_shift, self.frame_size, torch.hamming_window(self.frame_size).type_as(_mix_ri_spec), center=False)
		return mix_sig



	def on_test_batch_end(
		self,
		trainer: "pl.Trainer",
		pl_module: "pl.LightningModule",
		outputs: Optional[STEP_OUTPUT],
		batch: Any,
		batch_idx: int,
		dataloader_idx: int,
		) -> None:
		mix_ri_spec, tgt_ri_spec, doa_indices = batch
			
		if self.se_metrics_flag:
			est_ri_spec = outputs['est_ri_spec']  
			mix_sig = self.format_istft(mix_ri_spec)
			est_sig = self.format_istft(est_ri_spec)
			tgt_sig = self.format_istft(tgt_ri_spec)

			ref_ch_indices = [(mix_sig.shape[0]//est_sig.shape[0])*i for i in range(est_sig.shape[0])]
			mix_metrics = eval_metrics_batch_v1(tgt_sig.cpu().numpy(), mix_sig[ref_ch_indices,:].cpu().numpy())
			self.log("MIX_SNR", mix_metrics['snr'], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
			self.log("MIX_STOI", mix_metrics['stoi'], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
			self.log("MIX_ESTOI", mix_metrics['e_stoi'], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
			self.log("MIX_PESQ_NB", mix_metrics['pesq_nb'], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
			self.log("MIX_PESQ_WB", mix_metrics['pesq_wb'], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)


			_metrics = eval_metrics_batch_v1(tgt_sig.cpu().numpy(), est_sig.cpu().numpy())
			self.log("SNR", _metrics['snr'], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
			self.log("STOI", _metrics['stoi'], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
			self.log("ESTOI", _metrics['e_stoi'], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
			self.log("PESQ_NB", _metrics['pesq_nb'], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
			self.log("PESQ_WB", _metrics['pesq_wb'], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

		softmax = nn.Softmax(dim=2)
		est_doa_logits = outputs["est_doa_logits"]
		
		doa_prob = softmax(est_doa_logits)
		est_doa_idx = torch.argmax(doa_prob, dim=2)

		est_vad = 1*(doa_indices>=0)
		
		est_doa_vad = (est_doa_idx + 1000)*est_vad - 1000

		#non_vad frames diff = -1000 + 1 = 999
		diff = torch.abs(est_doa_vad - doa_indices)

		#circular distance for circular mics (0 to 360)
		diff = torch.where( (180 < diff)*(diff < 360), 360 - diff, diff)

		num_correct_doa_frms = torch.sum(torch.where(diff<=self.doa_tol, 1, 0), dim=1) 
		num_vad_frms = torch.sum(est_vad,1)
		#print(f"num_correct_doa_frms: {num_correct_doa_frms} num_vad_frms: {num_vad_frms} \n")

		acc = torch.div(num_correct_doa_frms, num_vad_frms) 
		
		low_perf_doa = acc < 0.80
		indices = low_perf_doa.nonzero()
		print(f"batch_idx: {batch_idx} acc < 80% indices {indices} acc: {acc[indices]} est_doa_vad: {est_doa_vad[indices]} doa_indices: {doa_indices[indices]}\n")

		print(f"batch_idx: {batch_idx} acc: {acc} est_doa_vad: {est_doa_vad} doa_indices: {doa_indices}\n")
		#breakpoint()
		avg_batch_acc = torch.nanmean(acc)

		self.log("acc", avg_batch_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

		return



		
		

