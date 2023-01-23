import torch 
import torchaudio
import torchaudio.transforms as T
from torch.nn import functional as F
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT

from typing import Any, Optional

from metrics import eval_metrics_batch_v1, _mag_spec_mask, gettdsignal_mag_spec_mask
from masked_gcc_phat import gcc_phat, compute_vad, gcc_phat_loc_orient, block_doa, blk_vad #, get_acc
from array_setup import array_setup_10cm_2mic
import numpy as np

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



class Losscallbacks(Callback):
	def __init__(self):
		self.frame_sie = 320
		self.frame_shift = 160    #TODO

	def on_test_batch_end(
		self,
		trainer: "pl.Trainer",
		pl_module: "pl.LightningModule",
		outputs: Optional[STEP_OUTPUT],
		batch: Any,
		batch_idx: int,
		dataloader_idx: int,
		) -> None:
		mix_ri_spec, tgt_ri_spec, doa = batch
		est_ri_spec = outputs['est_ri_spec']

		#adjusting shape, type for istft
		tgt_ri_spec = tgt_ri_spec.to(torch.float32)
		est_ri_spec = est_ri_spec.to(torch.float32)
		mix_ri_spec = mix_ri_spec.to(torch.float32)

		_mix_ri_spec = torch.permute(mix_ri_spec[:,:2],[0,3,2,1])
		mix_sig = torch.istft(_mix_ri_spec, 320,160,320,torch.hamming_window(320).type_as(_mix_ri_spec))

		_tgt_ri_spec = torch.permute(tgt_ri_spec,[0,3,2,1])
		tgt_sig = torch.istft(_tgt_ri_spec, 320,160,320,torch.hamming_window(320).type_as(_tgt_ri_spec))

		_est_ri_spec = torch.permute(est_ri_spec,[0,3,2,1])
		est_sig = torch.istft(_est_ri_spec, 320,160,320,torch.hamming_window(320).type_as(_est_ri_spec))



		#torch.rad2deg(doa[0,:,1])

		#torchaudio.save(f'mix_{batch_idx}.wav', (mix_sig/torch.max(torch.abs(mix_sig))).cpu(), sample_rate=16000)
		#torchaudio.save(f'tgt_{batch_idx}.wav', (tgt_sig/torch.max(torch.abs(tgt_sig))).cpu(), sample_rate=16000)
		#torchaudio.save(f'est_{batch_idx}.wav', (est_sig/torch.max(torch.abs(est_sig))).cpu(), sample_rate=16000)

		mix_metrics = eval_metrics_batch_v1(tgt_sig.cpu().numpy(), mix_sig.cpu().numpy())
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

		opinion = None
		opinion = "good" if (_metrics['stoi'] - mix_metrics['stoi']) > 0.2 else opinion
		opinion = "decent" if (_metrics['stoi'] - mix_metrics['stoi']) < 0.05 else opinion

		if opinion:
			trainer.logger.experiment.add_audio(f'mix_{batch_idx}_{opinion}', mix_sig/torch.max(torch.abs(mix_sig)), sample_rate=16000)
			trainer.logger.experiment.add_audio(f'tgt_{batch_idx}_{opinion}', tgt_sig/torch.max(torch.abs(tgt_sig)), sample_rate=16000)
			trainer.logger.experiment.add_audio(f'est_{batch_idx}_{opinion}', est_sig/torch.max(torch.abs(est_sig)), sample_rate=16000)

		return


	def on_validation_batch_end(
		self,
		trainer: "pl.Trainer",
		pl_module: "pl.LightningModule",
		outputs: Optional[STEP_OUTPUT],
		batch: Any,
		batch_idx: int,
		dataloader_idx: int,
		) -> None:
		mix_ri_spec, tgt_ri_spec, doa = batch
		est_ri_spec = outputs['est_ri_spec']

		#adjusting shape, type for istft
		tgt_ri_spec = tgt_ri_spec.to(torch.float32)
		est_ri_spec = est_ri_spec.to(torch.float32)
		mix_ri_spec = mix_ri_spec.to(torch.float32)

		_mix_ri_spec = torch.permute(mix_ri_spec[:,:2],[0,3,2,1])
		mix_sig = torch.istft(_mix_ri_spec, 320,160,320,torch.hamming_window(320).type_as(_mix_ri_spec))

		_tgt_ri_spec = torch.permute(tgt_ri_spec,[0,3,2,1])
		tgt_sig = torch.istft(_tgt_ri_spec, 320,160,320,torch.hamming_window(320).type_as(_tgt_ri_spec))

		_est_ri_spec = torch.permute(est_ri_spec,[0,3,2,1])
		est_sig = torch.istft(_est_ri_spec, 320,160,320,torch.hamming_window(320).type_as(_est_ri_spec))
		
		#print(f"mix val batch idx: {batch_idx} \n")
		if 0 == pl_module.current_epoch:
			breakpoint()
			mix_metrics = eval_metrics_batch_v1(tgt_sig.cpu().numpy(), mix_sig.cpu().numpy())
			"""
			self.log("MIX_SNR", mix_metrics['snr'], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
			self.log("MIX_STOI", mix_metrics['stoi'], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
			self.log("MIX_ESTOI", mix_metrics['e_stoi'], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
			self.log("MIX_PESQ_NB", mix_metrics['pesq_nb'], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
			self.log("MIX_PESQ_WB", mix_metrics['pesq_wb'], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
			"""
		#print(f"est val batch idx: {batch_idx} \n")
		breakpoint()
		_metrics = eval_metrics_batch_v1(tgt_sig.cpu().numpy(), est_sig.cpu().numpy())
		"""
		self.log("VAL_SNR", _metrics['snr'], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		self.log("VAL_STOI", _metrics['stoi'], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		self.log("VAL_ESTOI", _metrics['e_stoi'], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		self.log("VAL_PESQ_NB", _metrics['pesq_nb'], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		self.log("VAL_PESQ_WB", _metrics['pesq_wb'], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		"""
		return



class DOAcallbacks(Callback):
	def __init__(self, array_setup=array_setup_10cm_2mic, dataset_dtype=None, dataset_condition=None):
		self.frame_size = 320
		self.frame_shift = 160    #TODO
		#Computing DOA ( F -> T -> F) bcz of vad
		self.array_setup = array_setup
		self.local_mic_pos = torch.from_numpy(self.array_setup.mic_pos)
		self.local_mic_center=torch.from_numpy(np.array([0.0, 0.0, 0.0]))

		self.dataset_dtype= dataset_dtype
		self.dataset_condition = dataset_condition

	def format_istft(self, mix_ri_spec):
		#adjusting shape, type for istft
		#(batch_size, 4, T, F) . -> (batch_size, F, T, 2)
		mix_ri_spec = mix_ri_spec.to(torch.float32)

		(batch_size, n_ch, n_frms, n_freq) = mix_ri_spec.shape

		intm_mix_ri_spec = torch.reshape(mix_ri_spec, (batch_size*n_ch, n_frms, n_freq))
		intm_mix_ri_spec = torch.reshape(intm_mix_ri_spec, (intm_mix_ri_spec.shape[0]//2, 2, n_frms, n_freq))
		_mix_ri_spec = torch.permute(intm_mix_ri_spec,[0,3,2,1])

		mix_sig = torch.istft(_mix_ri_spec, self.frame_size,self.frame_shift,self.frame_size,torch.hamming_window(self.frame_size).type_as(_mix_ri_spec), center=False)
		return mix_sig

	def format_complex(self, mix_ri_spec):
		#adjusting shape, type for complex
		#(batch_size, 4, T, F) . -> (batch_size*2, T, F)
		mix_ri_spec = mix_ri_spec.to(torch.float32)

		(batch_size, n_ch, n_frms, n_freq) = mix_ri_spec.shape

		intm_mix_ri_spec = torch.reshape(mix_ri_spec, (batch_size*n_ch, n_frms, n_freq))
		intm_mix_ri_spec = torch.reshape(intm_mix_ri_spec, (intm_mix_ri_spec.shape[0]//2, 2, n_frms, n_freq))
		mix_ri_spec_cmplx = torch.complex(intm_mix_ri_spec[:,0,:,:], intm_mix_ri_spec[:,1,:,:])

		return mix_ri_spec_cmplx

	def get_acc(self, est_vad, est_blk_val, tgt_blk_val, tol=5, vad_th=0.6):
		n_blocks = len(est_vad)
		non_vad_blks =[]
		acc=0
		valid_blk_count = 0
		for idx in range(0, n_blocks):
			if est_vad[idx] >=vad_th:
				if (torch.abs(est_blk_val[idx] - torch.abs(tgt_blk_val[idx])) <= tol):
					acc += 1
				valid_blk_count +=1
			else:
				non_vad_blks.append(idx)

		acc /= valid_blk_count
		
		#print(f'n_blocks: {n_blocks}, non_vad_blks: {non_vad_blks}')
		return acc

	def _batch_metrics(self, batch, outputs):
		mix_ri_spec, tgt_ri_spec, doa = batch
		est_ri_spec = outputs['est_ri_spec']  #(b, n_ch, T, F)

		## SE metrics
		#adjusting shape, type for istft
		mix_sig = self.format_istft(mix_ri_spec)
		est_sig = self.format_istft(est_ri_spec)
		tgt_sig = self.format_istft(tgt_ri_spec)


		mix_metrics = eval_metrics_batch_v1(tgt_sig.cpu().numpy(), mix_sig.cpu().numpy())
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

		"""
		opinion = None
		opinion = "good" if (_metrics['stoi'] - mix_metrics['stoi']) > 0.2 else opinion
		opinion = "decent" if (_metrics['stoi'] - mix_metrics['stoi']) < 0.05 else opinion

		if opinion:
			trainer.logger.experiment.add_audio(f'mix_{batch_idx}_{opinion}', mix_sig/torch.max(torch.abs(mix_sig)), sample_rate=16000)
			trainer.logger.experiment.add_audio(f'tgt_{batch_idx}_{opinion}', tgt_sig/torch.max(torch.abs(tgt_sig)), sample_rate=16000)
			trainer.logger.experiment.add_audio(f'est_{batch_idx}_{opinion}', est_sig/torch.max(torch.abs(est_sig)), sample_rate=16000)

		"""


		#Vad 
		sig_vad_1 = compute_vad(tgt_sig[0,:].cpu().numpy(), self.frame_size, self.frame_shift)
		sig_vad_2 = compute_vad(tgt_sig[1,:].cpu().numpy(), self.frame_size, self.frame_shift)
		tgt_sig_vad = sig_vad_1*sig_vad_2
		tgt_sig_vad = tgt_sig_vad.to(device=tgt_sig.device)

		mix_spec_cmplx = self.format_complex(mix_ri_spec)
		tgt_spec_cmplx = self.format_complex(tgt_ri_spec)
		est_spec_cmplx = self.format_complex(est_ri_spec)

		mix_f_doa, mix_f_vals, mix_utt_doa, _, _ = gcc_phat_loc_orient(mix_spec_cmplx, torch.abs(mix_spec_cmplx), 16000, self.frame_size, self.local_mic_pos, self.local_mic_center, src_mic_dist=1, weighted=True, sig_vad=tgt_sig_vad, is_euclidean_dist=False)
		tgt_f_doa, tgt_f_vals, tgt_utt_doa, _, _ = gcc_phat_loc_orient(tgt_spec_cmplx, torch.abs(tgt_spec_cmplx), 16000, self.frame_size, self.local_mic_pos, self.local_mic_center, src_mic_dist=1, weighted=True, sig_vad=tgt_sig_vad, is_euclidean_dist=False)
		est_f_doa, est_f_vals, est_utt_doa, _, _ = gcc_phat_loc_orient(est_spec_cmplx, torch.abs(est_spec_cmplx), 16000, self.frame_size, self.local_mic_pos, self.local_mic_center, src_mic_dist=1, weighted=True, sig_vad=tgt_sig_vad, is_euclidean_dist=False)

		
		mix_frm_Acc = self.get_acc(tgt_sig_vad, mix_f_doa, tgt_f_doa, vad_th=0.6)
		est_frm_Acc = self.get_acc(tgt_sig_vad, est_f_doa, tgt_f_doa,vad_th=0.6)


		blk_size = 25
		mix_blk_vals = block_doa(frm_val=mix_f_vals, block_size=blk_size)
		tgt_blk_vals = block_doa(frm_val=tgt_f_vals, block_size=blk_size)
		est_blk_vals = block_doa(frm_val=est_f_vals, block_size=blk_size)

		#lbl_blk_doa, lbl_blk_range = block_lbl_doa(lbl_doa, block_size=blk_size)
		tgt_blk_vad = blk_vad(tgt_sig_vad, blk_size)
		#breakpoint()
		mix_Acc = self.get_acc(tgt_blk_vad, mix_blk_vals, tgt_blk_vals)
		est_Acc = self.get_acc(tgt_blk_vad, est_blk_vals, tgt_blk_vals)

		
		#print(mix_frm_Acc, est_frm_Acc, mix_Acc, est_Acc)

		self.log("mix_frm_Acc", mix_frm_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		self.log("est_frm_Acc", est_frm_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		self.log("mix_blk_Acc", mix_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		self.log("est_blk_Acc", est_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

		return


	def on_test_batch_end(
		self,
		trainer: "pl.Trainer",
		pl_module: "pl.LightningModule",
		outputs: Optional[STEP_OUTPUT],
		batch: Any,
		batch_idx: int,
		dataloader_idx: int,
		) -> None:

		self._batch_metrics(batch, outputs)
		"""
		pp_str = f'../signals/tr_s_test_{self.dataset_dtype}_{self.dataset_condition}_{app_str}'    # from_datasetfile_10sec/v2/'
				
		app_str = f'{batch_idx}'
		torch.save( {'mix': (mix_f_doa, mix_f_vals, mix_utt_doa),
						'tgt': (tgt_f_doa, tgt_f_vals, tgt_sig_vad, tgt_utt_doa),
						'est': (est_f_doa, est_f_vals, est_utt_doa),
						'lbl_doa': doa,
						'acc_metrics': (mix_frm_Acc, est_frm_Acc, mix_Acc, est_Acc)
						#'mix_metrics': mix_metrics,
						#'est_metrics': est_metrics 
						}, f'{pp_str}doa_{app_str}.pt',
						)
		"""


	def on_validation_batch_end(
		self,
		trainer: "pl.Trainer",
		pl_module: "pl.LightningModule",
		outputs: Optional[STEP_OUTPUT],
		batch: Any,
		batch_idx: int,
		dataloader_idx: int,
		) -> None:

		self._batch_metrics(batch, outputs)
