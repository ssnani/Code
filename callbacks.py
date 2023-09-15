import torch 
import torchaudio
import torchaudio.transforms as T
from torch.nn import functional as F
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT

from typing import Any, Optional

from metrics import eval_metrics_batch_v1, _mag_spec_mask, gettdsignal_mag_spec_mask, eval_metrics_v1
from masked_gcc_phat import gcc_phat, compute_vad, gcc_phat_loc_orient, block_doa, blk_vad, block_lbl_doa, gcc_phat_all_pairs, gcc_phat_all_pairs_masking #, get_acc
from array_setup import array_setup_10cm_2mic
import numpy as np
import os 
import csv

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
	def __init__(self, array_config, dataset_dtype=None, dataset_condition=None, noise_simulation = None, doa_tol = 5, doa_euclid_dist=False, mic_pairs=None, wgt_mech=None, loss_flags=None, log_str = None, dbg_doa_log = False):
		self.frame_size = 320
		self.frame_shift = 160    #TODO
		#Computing DOA ( F -> T -> F) bcz of vad
		#self.array_setup = array_setup
		self.array_config = array_config
		self.array_setup=array_config['array_setup']
		self.local_mic_pos = torch.from_numpy(self.array_setup.mic_pos)
		self.local_mic_center=torch.from_numpy(np.array([0.0, 0.0, 0.0]))

		self.dataset_dtype= dataset_dtype
		self.dataset_condition = dataset_condition
		self.noise_simulation = noise_simulation
		self.tol = doa_tol
		self.euclid_dist = doa_euclid_dist

		self.num_mics = self.local_mic_pos.shape[0]
		self.mic_pairs = mic_pairs #[(mic_1, mic_2) for mic_1 in range(0, self.num_mics) for mic_2 in range(mic_1+1, self.num_mics)]
		#self.acc = 0
		#self.files = 0
		self.wgt_mech = wgt_mech
		self.loss_flags = loss_flags
		self.log_str = log_str
		self.dbg_log = dbg_doa_log

	def get_mask(self, mix, est):
		noise = mix - est
		mask = torch.sqrt( torch.abs(est)**2/(torch.abs(noise)**2 + torch.abs(est)**2))
		return mask

	def format_istft(self, mix_ri_spec):
		#adjusting shape, type for istft
		#(batch_size, num_ch, T, F) . -> (batch_size*(num_ch/2), F, T, 2)
		mix_ri_spec = mix_ri_spec.to(torch.float32)

		(batch_size, n_ch, n_frms, n_freq) = mix_ri_spec.shape

		intm_mix_ri_spec = torch.reshape(mix_ri_spec, (batch_size*n_ch, n_frms, n_freq))
		intm_mix_ri_spec = torch.reshape(intm_mix_ri_spec, (intm_mix_ri_spec.shape[0]//2, 2, n_frms, n_freq))
		_mix_ri_spec = torch.permute(intm_mix_ri_spec,[0,3,2,1])

		mix_sig = torch.istft(_mix_ri_spec, self.frame_size,self.frame_shift,self.frame_size,torch.hamming_window(self.frame_size).type_as(_mix_ri_spec), center=False)
		return mix_sig

	def format_complex(self, mix_ri_spec):
		#adjusting shape, type for complex
		#(batch_size, num_ch, T, F) . -> (batch_size*(num_ch/2), T, F)
		mix_ri_spec = mix_ri_spec.to(torch.float32)

		(batch_size, n_ch, n_frms, n_freq) = mix_ri_spec.shape

		intm_mix_ri_spec = torch.reshape(mix_ri_spec, (batch_size*n_ch, n_frms, n_freq))
		intm_mix_ri_spec = torch.reshape(intm_mix_ri_spec, (intm_mix_ri_spec.shape[0]//2, 2, n_frms, n_freq))
		mix_ri_spec_cmplx = torch.complex(intm_mix_ri_spec[:,0,:,:], intm_mix_ri_spec[:,1,:,:])

		return mix_ri_spec_cmplx

	def get_acc(self, est_vad, est_blk_val, tgt_blk_val, tol=5, vad_th=0.6):
		n_blocks = len(est_vad)
		non_vad_blks =[]
		acc=0.0
		valid_blk_count = 0.0
		mae_only_correct_frms, mae_only_incorrect_frms, mae_overall_only_vad_frms = 0.0,0.0,0.0
		for idx in range(0, n_blocks):
			if est_vad[idx] >=vad_th:
				diff = torch.abs(est_blk_val[idx] - torch.abs(tgt_blk_val[idx]))
				if ( diff <= tol):
					acc += 1.0
					mae_only_correct_frms += diff
				else:
					mae_only_incorrect_frms += diff
				
				mae_overall_only_vad_frms += diff
				valid_blk_count += 1.0
			else:
				non_vad_blks.append(idx)
		if acc !=0:
			mae_only_correct_frms /= acc
		if (valid_blk_count-acc) !=0:
			mae_only_incorrect_frms /= (valid_blk_count-acc)

		mae_overall_only_vad_frms /= valid_blk_count

		acc /= valid_blk_count
		
		#print(f'n_blocks: {n_blocks}, non_vad_blks: {non_vad_blks}')
		return acc, mae_only_correct_frms, mae_only_incorrect_frms, mae_overall_only_vad_frms

	def _batch_metrics_masking(self, batch, outputs, batch_idx):
		_log_ps, _tgt_mask, seq_len, doa, mix_cs = batch
		est_mask = outputs['est_masks']  #(n_ch, T, F)
		
		#mix_cs = mix_cs.squeeze()
		_tgt_mask = _tgt_mask.squeeze()
		#breakpoint()
		est_ri_spec = torch.repeat_interleave(est_mask,2,dim=0).unsqueeze(dim=0)*mix_cs
		tgt_ri_spec = torch.repeat_interleave(_tgt_mask,2,dim=0).unsqueeze(dim=0)*mix_cs

		mix_sig = self.format_istft(mix_cs)
		est_sig = self.format_istft(est_ri_spec)
		tgt_sig = self.format_istft(tgt_ri_spec)

		mix_sig = mix_sig#/torch.max(torch.abs(mix_sig))
		tgt_sig = tgt_sig#/torch.max(torch.abs(tgt_sig))
		est_sig = est_sig#/torch.max(torch.abs(est_sig))

		"""
		torchaudio.save(f'../dbg_signals/simu_rirs_test/mix_{batch_idx}_mvn_false.wav', (mix_sig/torch.max(torch.abs(mix_sig))).cpu(), sample_rate=16000)
		torchaudio.save(f'../dbg_signals/simu_rirs_test/tgt_{batch_idx}_mvn_false.wav', (tgt_sig/torch.max(torch.abs(tgt_sig))).cpu(), sample_rate=16000)
		torchaudio.save(f'../dbg_signals/simu_rirs_test/est_{batch_idx}_mvn_false.wav', (est_sig/torch.max(torch.abs(est_sig))).cpu(), sample_rate=16000)
		"""

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


		#Vad 
		vad_comp_sig = tgt_sig/torch.max(torch.abs(tgt_sig)) if "real_rirs" not in self.array_config else est_sig

		num_frms =  (vad_comp_sig.shape[1]-self.frame_size +1)//self.frame_shift + 1
		
		tgt_sig_vad = torch.ones(num_frms)
		for idx in range(self.num_mics):
			sig_vad = compute_vad(vad_comp_sig[idx,:].cpu().numpy(), self.frame_size, self.frame_shift)
			tgt_sig_vad *= sig_vad 
			
		tgt_sig_vad = tgt_sig_vad.to(device=tgt_sig.device)


		mix_spec_cmplx = self.format_complex(mix_cs)

		est_8mic_f_doa, tgt_8mic_f_doa, mix_8mic_f_doa, est_8mic_utt_doa, tgt_8mic_utt_doa, mix_8mic_utt_doa, est_2mic_f_doa, tgt_2mic_f_doa, mix_2mic_f_doa, est_2mic_utt_doa, tgt_2mic_utt_doa, mix_2mic_utt_doa= gcc_phat_all_pairs_masking(mix_spec_cmplx, est_mask, _tgt_mask, 16000, self.frame_size, self.local_mic_pos, 
								 										self.local_mic_center, src_mic_dist=1, weighted=True, sig_vad=tgt_sig_vad, is_euclidean_dist=self.euclid_dist, mic_pairs=self.mic_pairs)
		
		
		
		if "real_rirs" not in self.array_config:
			ref_8mic_f_doa = tgt_8mic_f_doa
			ref_2mic_f_doa = tgt_2mic_f_doa
		else:
			ref_f_doa = torch.repeat(doa, mix_8mic_f_doa.shape[0]) #torch.rad2deg(doa[:,:,-1])[0,:mix_f_doa.shape[0]]

		mix_8mic_frm_Acc, mix_8mic_mae_only_correct_frms, mix_8mic_mae_only_incorrect_frms, mix_8mic_mae_overall_only_vad_frms = self.get_acc(tgt_sig_vad, mix_8mic_f_doa, ref_8mic_f_doa, tol=self.tol, vad_th=0.6)
		mix_2mic_frm_Acc, mix_2mic_mae_only_correct_frms, mix_2mic_mae_only_incorrect_frms, mix_2mic_mae_overall_only_vad_frms = self.get_acc(tgt_sig_vad, mix_2mic_f_doa, ref_2mic_f_doa, tol=self.tol, vad_th=0.6)

		est_8mic_frm_Acc, est_8mic_mae_only_correct_frms, est_8mic_mae_only_incorrect_frms, est_8mic_mae_overall_only_vad_frms = self.get_acc(tgt_sig_vad, est_8mic_f_doa, ref_8mic_f_doa, tol=self.tol, vad_th=0.6)
		est_2mic_frm_Acc, est_2mic_mae_only_correct_frms, est_2mic_mae_only_incorrect_frms, est_2mic_mae_overall_only_vad_frms = self.get_acc(tgt_sig_vad, est_2mic_f_doa, ref_2mic_f_doa, tol=self.tol, vad_th=0.6)

		#Utterance level
		mix_8mic_utt_Acc = 1.0 if torch.abs(mix_8mic_utt_doa-tgt_8mic_utt_doa) <= 5 else 0
		est_8mic_utt_Acc = 1.0 if torch.abs(est_8mic_utt_doa-tgt_8mic_utt_doa) <= 5 else 0

		mix_2mic_utt_Acc = 1.0 if torch.abs(mix_2mic_utt_doa-tgt_2mic_utt_doa) <= 5 else 0
		est_2mic_utt_Acc = 1.0 if torch.abs(est_2mic_utt_doa-tgt_2mic_utt_doa) <= 5 else 0

		print(batch_idx, mix_8mic_frm_Acc, est_8mic_frm_Acc, mix_2mic_frm_Acc, est_2mic_frm_Acc, mix_8mic_utt_Acc, est_8mic_utt_Acc, mix_2mic_utt_Acc, est_2mic_utt_Acc)

		self.log("mix_8mic_frm_Acc", mix_8mic_frm_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		self.log("est_8mic_frm_Acc", est_8mic_frm_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		self.log("mix_2mic_frm_Acc", mix_2mic_frm_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		self.log("est_2mic_frm_Acc", est_2mic_frm_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

		self.log("mix_8mic_utt_Acc", mix_8mic_utt_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		self.log("est_8mic_utt_Acc", est_8mic_utt_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		self.log("mix_2mic_utt_Acc", mix_2mic_utt_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		self.log("est_2mic_utt_Acc", est_2mic_utt_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		

		self.log("mix_8mic_mae_overall_only_vad_frms", mix_8mic_mae_overall_only_vad_frms, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		self.log("mix_2mic_mae_overall_only_vad_frms", mix_2mic_mae_overall_only_vad_frms, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		self.log("est_8mic_mae_overall_only_vad_frms", est_8mic_mae_overall_only_vad_frms, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		self.log("est_2mic_mae_overall_only_vad_frms", est_2mic_mae_overall_only_vad_frms, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

		self.log("mix_8mic_mae_only_correct_frms", mix_8mic_mae_only_correct_frms, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		self.log("mix_2mic_mae_only_correct_frms", mix_2mic_mae_only_correct_frms, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		self.log("est_8mic_mae_only_correct_frms", est_8mic_mae_only_correct_frms, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		self.log("est_2mic_mae_only_correct_frms", est_2mic_mae_only_correct_frms, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

		self.log("mix_8mic_mae_only_incorrect_frms", mix_8mic_mae_only_incorrect_frms, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		self.log("mix_2mic_mae_only_incorrect_frms", mix_2mic_mae_only_incorrect_frms, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		self.log("est_8mic_mae_only_incorrect_frms", est_8mic_mae_only_incorrect_frms, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		self.log("est_2mic_mae_only_incorrect_frms", est_2mic_mae_only_incorrect_frms, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		return


	def _batch_metrics(self, batch, outputs, batch_idx):
		mix_ri_spec, tgt_ri_spec, doa = batch
		est_ri_spec = outputs['est_ri_spec']  #(b, n_ch, T, F)

		## SE metrics
		#adjusting shape, type for istft
		mix_sig = self.format_istft(mix_ri_spec)
		est_sig = self.format_istft(est_ri_spec)
		tgt_sig = self.format_istft(tgt_ri_spec)

		#normalize the signals --- observed improves webrtc vad 
		mix_sig = mix_sig/torch.max(torch.abs(mix_sig))
		tgt_sig = tgt_sig/torch.max(torch.abs(tgt_sig))
		est_sig = est_sig/torch.max(torch.abs(est_sig))
		"""
		torchaudio.save(f'../signals/real_rirs_dbg/mix_{batch_idx}.wav', (mix_sig/torch.max(torch.abs(mix_sig))).cpu(), sample_rate=16000)
		torchaudio.save(f'../signals/real_rirs_dbg/tgt_{batch_idx}.wav', (tgt_sig/torch.max(torch.abs(tgt_sig))).cpu(), sample_rate=16000)
		torchaudio.save(f'../signals/real_rirs_dbg/est_{batch_idx}.wav', (est_sig/torch.max(torch.abs(est_sig))).cpu(), sample_rate=16000)
		"""
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
		vad_comp_sig = tgt_sig if "real_rirs" not in self.array_config else est_sig
		
		sig_vad_1 = compute_vad(vad_comp_sig[0,:].cpu().numpy(), self.frame_size, self.frame_shift)
		sig_vad_2 = compute_vad(vad_comp_sig[1,:].cpu().numpy(), self.frame_size, self.frame_shift)
		tgt_sig_vad = sig_vad_1*sig_vad_2
		tgt_sig_vad = tgt_sig_vad.to(device=tgt_sig.device)


		mix_spec_cmplx = self.format_complex(mix_ri_spec)
		tgt_spec_cmplx = self.format_complex(tgt_ri_spec)
		est_spec_cmplx = self.format_complex(est_ri_spec)

		#removing spatial aliasing freq bins


		mix_f_doa, mix_f_vals, mix_utt_doa, _, _ = gcc_phat_loc_orient(mix_spec_cmplx, torch.abs(mix_spec_cmplx), 16000, self.frame_size, self.local_mic_pos, 
								 										self.local_mic_center, src_mic_dist=1, weighted=True, sig_vad=tgt_sig_vad, is_euclidean_dist=self.euclid_dist)
		tgt_f_doa, tgt_f_vals, tgt_utt_doa, tgt_unwieghted_freq_vals, tgt_freq_vals = gcc_phat_loc_orient(tgt_spec_cmplx, torch.abs(tgt_spec_cmplx), 16000, self.frame_size, self.local_mic_pos, 
								 										self.local_mic_center, src_mic_dist=1, weighted=True, sig_vad=tgt_sig_vad, is_euclidean_dist=self.euclid_dist)
		est_f_doa, est_f_vals, est_utt_doa, est_unwieghted_freq_vals, est_freq_vals = gcc_phat_loc_orient(est_spec_cmplx, torch.abs(est_spec_cmplx), 16000, self.frame_size, self.local_mic_pos, 
								 										self.local_mic_center, src_mic_dist=1, weighted=True, sig_vad=tgt_sig_vad, is_euclidean_dist=self.euclid_dist)

		if "real_rirs" not in self.array_config:
			ref_f_doa = tgt_f_doa
		else:
			ref_f_doa = torch.rad2deg(doa[:,:,-1])[0,:mix_f_doa.shape[0]]
		
		mix_frm_Acc,_,_,_ = self.get_acc(tgt_sig_vad, mix_f_doa, ref_f_doa, tol=self.tol, vad_th=0.6)
		est_frm_Acc,_,_,_ = self.get_acc(tgt_sig_vad, est_f_doa, ref_f_doa, tol=self.tol, vad_th=0.6)


		blk_size = 25
		mix_blk_vals = block_doa(frm_val=mix_f_vals, block_size=blk_size)
		tgt_blk_vals = block_doa(frm_val=tgt_f_vals, block_size=blk_size)
		est_blk_vals = block_doa(frm_val=est_f_vals, block_size=blk_size)

		#lbl_blk_doa, lbl_blk_range = block_lbl_doa(lbl_doa, block_size=blk_size)
		tgt_blk_vad = blk_vad(tgt_sig_vad, blk_size)
		#breakpoint()
		if "real_rirs" not in self.array_config:
			ref_blk_vals = tgt_blk_vals
		else:
			ref_blk_vals, _ = block_lbl_doa(doa, block_size=blk_size)#block_doa(frm_val=ref_f_doa, block_size=blk_size)

		mix_Acc,_,_,_ = self.get_acc(tgt_blk_vad, mix_blk_vals, ref_blk_vals, tol=self.tol)
		est_Acc,_,_,_ = self.get_acc(tgt_blk_vad, est_blk_vals, ref_blk_vals, tol=self.tol)

		#Utterance level
		doa_degrees = torch.abs(torch.rad2deg(doa[:,:,-1])[0,0])
		mix_utt_Acc = 1 if torch.abs(mix_utt_doa-doa_degrees) <= self.tol else 0 #tgt_utt_doa
		est_utt_Acc = 1 if torch.abs(est_utt_doa-doa_degrees) <= self.tol else 0 #tgt_utt_doa

		
		self.log("mix_frm_Acc", mix_frm_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		self.log("est_frm_Acc", est_frm_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		self.log("mix_blk_Acc", mix_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		self.log("est_blk_Acc", est_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		self.log("mix_utt_Acc", mix_utt_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		self.log("est_utt_Acc", est_utt_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		
		#print(batch_idx, mix_frm_Acc, est_frm_Acc, mix_Acc, est_Acc)
		print(batch_idx, mix_frm_Acc, est_frm_Acc, doa_degrees, mix_utt_doa, est_utt_doa, mix_utt_Acc, est_utt_Acc)
		
		#don't care for simulated rirs
		if "real_rirs" in self.array_config and ( not (doa_degrees==0 or doa_degrees==180)):

			#self.acc += est_utt_Acc
			#self.files += 1
			#print(batch_idx, self.acc, self.files)

			self.log("mix_frm_Acc_excluded_endfire", mix_frm_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
			self.log("est_frm_Acc_excluded_endfire", est_frm_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
			self.log("mix_blk_Acc_excluded_endfire", mix_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
			self.log("est_blk_Acc_excluded_endfire", est_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
			self.log("mix_utt_Acc_excluded_endfire", mix_utt_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
			self.log("est_utt_Acc_excluded_endfire", est_utt_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		
		"""
		torch.save({'mix': (mix_f_doa, mix_f_vals, mix_utt_doa),
					'tgt': (tgt_f_doa, tgt_f_vals, tgt_sig_vad, tgt_utt_doa, tgt_unwieghted_freq_vals, tgt_freq_vals),
					'est': (est_f_doa, est_f_vals, est_utt_doa, est_unwieghted_freq_vals, est_freq_vals),
					'doa': doa }, 
					f'../signals/simu_rirs_dbg/doa_{1.0}_{batch_idx}_tol_{self.tol}deg_euclid_{self.euclid_dist}_rir_dp_t60_0_train_MIMO_RI_PD_mag_compression.pt', 
			)
		"""
		return
	
	def _batch_metrics_v2(self, batch, outputs, batch_idx):
		mix_ri_spec, tgt_ri_spec, doa = batch
		est_ri_spec = outputs['est_ri_spec']  #(b, n_ch, T, F)

		## SE metrics
		#adjusting shape, type for istft
		mix_sig = self.format_istft(mix_ri_spec)
		est_sig = self.format_istft(est_ri_spec)
		tgt_sig = self.format_istft(tgt_ri_spec)

		#normalize the signals --- observed improves webrtc vad 
		mix_sig = mix_sig/torch.max(torch.abs(mix_sig))
		tgt_sig = tgt_sig/torch.max(torch.abs(tgt_sig))
		est_sig = est_sig/torch.max(torch.abs(est_sig))
		if self.dbg_log:
			torchaudio.save(f'../signals/simu_rirs_dbg/mix_{0.2}_ind_16bitproc_{batch_idx}.wav', (mix_sig/torch.max(torch.abs(mix_sig))).cpu(), sample_rate=16000)
			torchaudio.save(f'../signals/simu_rirs_dbg/tgt_{0.2}_ind_16bitproc_{batch_idx}.wav', (tgt_sig/torch.max(torch.abs(tgt_sig))).cpu(), sample_rate=16000)
			torchaudio.save(f'../signals/simu_rirs_dbg/est_{0.2}_ind_16bitproc_{batch_idx}.wav', (est_sig/torch.max(torch.abs(est_sig))).cpu(), sample_rate=16000)
		
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
		vad_comp_sig = tgt_sig if "real_rirs" not in self.array_config else est_sig

		num_frms =  (vad_comp_sig.shape[1]-self.frame_size +1)//self.frame_shift + 1
		
		tgt_sig_vad = torch.ones(num_frms)
		for idx in range(self.num_mics):
			sig_vad = compute_vad(vad_comp_sig[idx,:].cpu().numpy(), self.frame_size, self.frame_shift)
			tgt_sig_vad *= sig_vad 
			
		tgt_sig_vad = tgt_sig_vad.to(device=tgt_sig.device)


		mix_spec_cmplx = self.format_complex(mix_ri_spec)
		tgt_spec_cmplx = self.format_complex(tgt_ri_spec)
		est_spec_cmplx = self.format_complex(est_ri_spec)

		if self.wgt_mech=="MASK":
			#mask
			mix_wt = self.get_mask(mix_spec_cmplx, mix_spec_cmplx)
			tgt_wt = self.get_mask(mix_spec_cmplx, tgt_spec_cmplx)
			est_wt = self.get_mask(mix_spec_cmplx, est_spec_cmplx)
		else:
			mix_wt = torch.abs(mix_spec_cmplx)
			tgt_wt = torch.abs(tgt_spec_cmplx)
			est_wt = torch.abs(est_spec_cmplx)


		mix_f_doa, mix_f_vals, mix_utt_doa, mix_2mic_f_doa, mix_2mic_utt_doa, _ = gcc_phat_all_pairs(mix_spec_cmplx, mix_wt, 16000, self.frame_size, self.local_mic_pos, 
								 										self.local_mic_center, src_mic_dist=1, weighted=True, sig_vad=tgt_sig_vad, is_euclidean_dist=self.euclid_dist, mic_pairs=self.mic_pairs)
		tgt_f_doa, tgt_f_vals, tgt_utt_doa, tgt_2mic_f_doa, tgt_2mic_utt_doa, _ = gcc_phat_all_pairs(tgt_spec_cmplx, tgt_wt, 16000, self.frame_size, self.local_mic_pos, 
								 										self.local_mic_center, src_mic_dist=1, weighted=True, sig_vad=tgt_sig_vad, is_euclidean_dist=self.euclid_dist, mic_pairs=self.mic_pairs)
		est_f_doa, est_f_vals, est_utt_doa, est_2mic_f_doa, est_2mic_utt_doa, _ = gcc_phat_all_pairs(est_spec_cmplx, est_wt, 16000, self.frame_size, self.local_mic_pos, 
								 										self.local_mic_center, src_mic_dist=1, weighted=True, sig_vad=tgt_sig_vad, is_euclidean_dist=self.euclid_dist, mic_pairs=self.mic_pairs)

		if "real_rirs" not in self.array_config:
			ref_f_doa = tgt_f_doa
			ref_2mic_f_doa = tgt_2mic_f_doa
		else:
			ref_f_doa = torch.rad2deg(doa[:,:,-1])[0,:mix_f_doa.shape[0]]
			ref_2mic_f_doa = ref_f_doa
		
		mix_frm_Acc,_,_,_ = self.get_acc(tgt_sig_vad, mix_f_doa, ref_f_doa, tol=self.tol, vad_th=0.6)
		est_frm_Acc,_,_,_ = self.get_acc(tgt_sig_vad, est_f_doa, ref_f_doa, tol=self.tol, vad_th=0.6)

		mix_2mic_frm_Acc,_,_,_ = self.get_acc(tgt_sig_vad, mix_2mic_f_doa, ref_2mic_f_doa, tol=self.tol, vad_th=0.6)
		est_2mic_frm_Acc,_,_,_ = self.get_acc(tgt_sig_vad, est_2mic_f_doa, ref_2mic_f_doa, tol=self.tol, vad_th=0.6)

		blk_size = 25
		mix_blk_vals = block_doa(frm_val=mix_f_vals, block_size=blk_size)
		tgt_blk_vals = block_doa(frm_val=tgt_f_vals, block_size=blk_size)
		est_blk_vals = block_doa(frm_val=est_f_vals, block_size=blk_size)

		#lbl_blk_doa, lbl_blk_range = block_lbl_doa(lbl_doa, block_size=blk_size)
		tgt_blk_vad = blk_vad(tgt_sig_vad, blk_size)
		#breakpoint()
		if "real_rirs" not in self.array_config:
			ref_blk_vals = tgt_blk_vals
		else:
			ref_blk_vals, _ = block_lbl_doa(doa, block_size=blk_size)#block_doa(frm_val=ref_f_doa, block_size=blk_size)

		mix_Acc,_,_,_ = self.get_acc(tgt_blk_vad, mix_blk_vals, ref_blk_vals, tol=self.tol)
		est_Acc,_,_,_ = self.get_acc(tgt_blk_vad, est_blk_vals, ref_blk_vals, tol=self.tol)

		#Utterance level
		doa_degrees = torch.abs(torch.rad2deg(doa[:,:,-1])[0,0])
		mix_utt_Acc = 1.0 if torch.abs(mix_utt_doa-doa_degrees) <= self.tol else 0 #tgt_utt_doa
		est_utt_Acc = 1.0 if torch.abs(est_utt_doa-doa_degrees) <= self.tol else 0 #tgt_utt_doa

		mix_2mic_utt_Acc = 1.0 if torch.abs(mix_2mic_utt_doa-doa_degrees) <= self.tol else 0 #tgt_utt_doa
		est_2mic_utt_Acc = 1.0 if torch.abs(est_2mic_utt_doa-doa_degrees) <= self.tol else 0 #tgt_utt_doa
		
		self.log("mix_frm_Acc", mix_frm_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		self.log("est_frm_Acc", est_frm_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		self.log("mix_blk_Acc", mix_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		self.log("est_blk_Acc", est_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		self.log("mix_utt_Acc", mix_utt_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		self.log("est_utt_Acc", est_utt_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)


		self.log("mix_2mic_frm_Acc", mix_2mic_frm_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		self.log("est_2mic_frm_Acc", est_2mic_frm_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		self.log("mix_2mic_utt_Acc", mix_2mic_utt_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		self.log("est_2mic_utt_Acc", est_2mic_utt_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		#print(batch_idx, mix_frm_Acc, est_frm_Acc, mix_Acc, est_Acc)
		print(batch_idx, mix_frm_Acc, est_frm_Acc, doa_degrees, mix_utt_doa, est_utt_doa, mix_utt_Acc, est_utt_Acc)
		print(batch_idx, mix_2mic_frm_Acc, est_2mic_frm_Acc, doa_degrees, mix_2mic_utt_doa, est_2mic_utt_doa, mix_2mic_utt_Acc, est_2mic_utt_Acc)
		
		#don't care for simulated rirs
		if "real_rirs" in self.array_config and ( not (doa_degrees==0 or doa_degrees==180)):

			#self.acc += est_utt_Acc
			#self.files += 1
			#print(batch_idx, self.acc, self.files)

			self.log("mix_frm_Acc_excluded_endfire", mix_frm_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
			self.log("est_frm_Acc_excluded_endfire", est_frm_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
			self.log("mix_blk_Acc_excluded_endfire", mix_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
			self.log("est_blk_Acc_excluded_endfire", est_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
			self.log("mix_utt_Acc_excluded_endfire", mix_utt_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
			self.log("est_utt_Acc_excluded_endfire", est_utt_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

			self.log("mix_2mic_frm_Acc_excluded_endfire", mix_2mic_frm_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
			self.log("est_2mic_frm_Acc_excluded_endfire", est_2mic_frm_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
			self.log("mix_2mic_utt_Acc_excluded_endfire", mix_2mic_utt_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
			self.log("est_2mic_utt_Acc_excluded_endfire", est_2mic_utt_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		
		if self.dbg_log:
			torch.save({'mix': (mix_f_doa, mix_f_vals, mix_utt_doa),
						'tgt': (tgt_f_doa, tgt_f_vals, tgt_sig_vad, tgt_utt_doa),#, tgt_unwieghted_freq_vals, tgt_freq_vals),
						'est': (est_f_doa, est_f_vals, est_utt_doa),# est_unwieghted_freq_vals, est_freq_vals),
						'doa': doa }, 
						f'../signals/real_rirs_dbg/doa_{0.61}_{batch_idx}_tol_{self.tol}deg_euclid_{self.euclid_dist}_2mic_rir_dp_t60_0_train_MIMO_RI_PD_mag_compression.pt', 
				)
		
		return
	
	# for multiple loss functions testing
	def _batch_metrics_v3(self, batch, outputs, batch_idx):
		dbg_log = False

		mix_ri_spec, tgt_ri_spec, doa = batch
		est_ri_spec_all = outputs['est_ri_spec']  #(b, n_ch, T, F)

		num_batches = est_ri_spec_all.shape[0]
		num_mics = est_ri_spec_all.shape[1]//2
		## SE metrics
		#adjusting shape, type for istft
		mix_sig = self.format_istft(mix_ri_spec)
		est_sig_all = self.format_istft(est_ri_spec_all)
		tgt_sig = self.format_istft(tgt_ri_spec)

		#normalize the signals --- observed improves webrtc vad 
		mix_sig = mix_sig#/torch.max(torch.abs(mix_sig))
		tgt_sig = tgt_sig#/torch.max(torch.abs(tgt_sig))

		est_sig_all = est_sig_all #/torch.max(torch.abs(est_sig_all))
		if self.dbg_log:
			torchaudio.save(f'../signals/simu_rirs_dbg/loss_functions_comparison_2mic_reverb/mix_{0.2}_mimo_fwk_{batch_idx}.wav', (mix_sig/torch.max(torch.abs(mix_sig))).cpu(), sample_rate=16000)
			torchaudio.save(f'../signals/simu_rirs_dbg/loss_functions_comparison_2mic_reverb/tgt_{0.2}_mimo_fwk_{batch_idx}.wav', (tgt_sig/torch.max(torch.abs(tgt_sig))).cpu(), sample_rate=16000)
			
		mix_metrics = eval_metrics_batch_v1(tgt_sig.cpu().numpy(), mix_sig.cpu().numpy())
		self.log("MIX_SNR", mix_metrics['snr'], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		self.log("MIX_STOI", mix_metrics['stoi'], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		self.log("MIX_ESTOI", mix_metrics['e_stoi'], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		self.log("MIX_PESQ_NB", mix_metrics['pesq_nb'], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		self.log("MIX_PESQ_WB", mix_metrics['pesq_wb'], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)


		#Vad 
		vad_comp_sig = tgt_sig/torch.max(torch.abs(tgt_sig)) if "real_rirs" not in self.array_config else est_sig

		num_frms =  (vad_comp_sig.shape[1]-self.frame_size +1)//self.frame_shift + 1
		
		tgt_sig_vad = torch.ones(num_frms)
		for idx in range(self.num_mics):
			sig_vad = compute_vad(vad_comp_sig[idx,:].cpu().numpy(), self.frame_size, self.frame_shift)
			tgt_sig_vad *= sig_vad 
			
		tgt_sig_vad = tgt_sig_vad.to(device=tgt_sig.device)

		mix_spec_cmplx = self.format_complex(mix_ri_spec)
		tgt_spec_cmplx = self.format_complex(tgt_ri_spec)



		if self.wgt_mech=="MASK":
			#mask
			mix_wt = self.get_mask(mix_spec_cmplx, mix_spec_cmplx)
			tgt_wt = self.get_mask(mix_spec_cmplx, tgt_spec_cmplx)
		else:
			#print("DOA_MAG_WT",  self.local_mic_pos)
			mix_wt = torch.abs(mix_spec_cmplx)
			tgt_wt = torch.abs(tgt_spec_cmplx)
			

		mix_f_doa, mix_f_vals, mix_utt_doa, mix_2mic_f_doa, mix_2mic_utt_doa, _ = gcc_phat_all_pairs(mix_spec_cmplx, mix_wt, 16000, self.frame_size, self.local_mic_pos, 
								 										self.local_mic_center, src_mic_dist=1, weighted=True, sig_vad=tgt_sig_vad, is_euclidean_dist=self.euclid_dist, mic_pairs=self.mic_pairs)
		tgt_f_doa, tgt_f_vals, tgt_utt_doa, tgt_2mic_f_doa, tgt_2mic_utt_doa, _ = gcc_phat_all_pairs(tgt_spec_cmplx, tgt_wt, 16000, self.frame_size, self.local_mic_pos, 
								 										self.local_mic_center, src_mic_dist=1, weighted=True, sig_vad=tgt_sig_vad, is_euclidean_dist=self.euclid_dist, mic_pairs=self.mic_pairs)
		
		if "real_rirs" not in self.array_config:
			ref_f_doa = tgt_f_doa
			ref_2mic_f_doa = tgt_2mic_f_doa
		else:
			ref_f_doa = torch.rad2deg(doa[:,:,-1])[0,:mix_f_doa.shape[0]]
			ref_2mic_f_doa = ref_f_doa
		
		mix_frm_Acc, mix_mae_only_correct_frms, mix_mae_only_incorrect_frms, mix_mae_overall_only_vad_frms = self.get_acc(tgt_sig_vad, mix_f_doa, ref_f_doa, tol=self.tol, vad_th=0.6)
		mix_2mic_frm_Acc, mix_2mic_mae_only_correct_frms, mix_2mic_mae_only_incorrect_frms, mix_2mic_mae_overall_only_vad_frms = self.get_acc(tgt_sig_vad, mix_2mic_f_doa, ref_2mic_f_doa, tol=self.tol, vad_th=0.6)

		print('mix', mix_mae_only_correct_frms, mix_mae_only_incorrect_frms, mix_mae_overall_only_vad_frms, 'mix_2mic', mix_2mic_mae_only_correct_frms, mix_2mic_mae_only_incorrect_frms, mix_2mic_mae_overall_only_vad_frms)

		doa_degrees = torch.abs(torch.rad2deg(doa[:,:,-1])[0,0])
		mix_utt_Acc = 1.0 if torch.abs(mix_utt_doa-doa_degrees) <= self.tol else 0 #tgt_utt_doa
		mix_2mic_utt_Acc = 1.0 if torch.abs(mix_2mic_utt_doa-doa_degrees) <= self.tol else 0 #tgt_utt_doa

		self.log(f"mix_frm_Acc", mix_frm_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		self.log(f"mix_2mic_frm_Acc", mix_2mic_frm_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		self.log(f"mix_utt_Acc", mix_utt_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		self.log(f"mix_2mic_utt_Acc", mix_2mic_utt_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		
		self.log(f"mix_mae_only_correct_frms", mix_mae_only_correct_frms, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		self.log(f"mix_mae_only_incorrect_frms", mix_mae_only_incorrect_frms, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		self.log(f"mix_mae_overall_only_vad_frms", mix_mae_overall_only_vad_frms, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

		self.log(f"mix_2mic_mae_only_correct_frms", mix_2mic_mae_only_correct_frms, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		self.log(f"mix_2mic_mae_only_incorrect_frms", mix_2mic_mae_only_incorrect_frms, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		self.log(f"mix_2mic_mae_overall_only_vad_frms", mix_2mic_mae_overall_only_vad_frms, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

		for idx in range(num_batches):
			loss_flag = self.loss_flags[idx]
			est_sig = est_sig_all[idx*num_mics:idx*num_mics+num_mics,:]
			est_sig = est_sig#/torch.max(torch.abs(est_sig))
			if self.dbg_log:
				torchaudio.save(f'../signals/simu_rirs_dbg/loss_functions_comparison_2mic_reverb/est_{0.2}_mimo_fwk_{batch_idx}_{idx}_{loss_flag}.wav', (est_sig/torch.max(torch.abs(est_sig))).cpu(), sample_rate=16000)

			est_ri_spec = est_ri_spec_all[[idx]]
			
			_metrics = eval_metrics_batch_v1(tgt_sig.cpu().numpy(), est_sig.cpu().numpy())
			self.log(f"{loss_flag}_SNR", _metrics['snr'], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
			self.log(f"{loss_flag}_STOI", _metrics['stoi'], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
			self.log(f"{loss_flag}_ESTOI", _metrics['e_stoi'], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
			self.log(f"{loss_flag}_PESQ_NB", _metrics['pesq_nb'], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
			self.log(f"{loss_flag}_PESQ_WB", _metrics['pesq_wb'], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

			"""
			opinion = None
			opinion = "good" if (_metrics['stoi'] - mix_metrics['stoi']) > 0.2 else opinion
			opinion = "decent" if (_metrics['stoi'] - mix_metrics['stoi']) < 0.05 else opinion

			if opinion:
				trainer.logger.experiment.add_audio(f'mix_{batch_idx}_{opinion}', mix_sig/torch.max(torch.abs(mix_sig)), sample_rate=16000)
				trainer.logger.experiment.add_audio(f'tgt_{batch_idx}_{opinion}', tgt_sig/torch.max(torch.abs(tgt_sig)), sample_rate=16000)
				trainer.logger.experiment.add_audio(f'est_{batch_idx}_{opinion}', est_sig/torch.max(torch.abs(est_sig)), sample_rate=16000)

			"""

			est_spec_cmplx = self.format_complex(est_ri_spec)

			#removing spatial aliasing freq bins
			#print(mix_spec_cmplx.shape, tgt_spec_cmplx.shape, est_spec_cmplx.shape)
			if self.wgt_mech=="MASK":
			#mask
				est_wt = self.get_mask(mix_spec_cmplx, est_spec_cmplx)
			else:
				est_wt = torch.abs(est_spec_cmplx)
			
			est_f_doa, est_f_vals, est_utt_doa, est_2mic_f_doa, est_2mic_utt_doa, _ = gcc_phat_all_pairs(est_spec_cmplx, est_wt, 16000, self.frame_size, self.local_mic_pos, 
								 										self.local_mic_center, src_mic_dist=1, weighted=True, sig_vad=tgt_sig_vad, is_euclidean_dist=self.euclid_dist, mic_pairs=self.mic_pairs)

			
			est_frm_Acc, est_mae_only_correct_frms, est_mae_only_incorrect_frms, est_mae_overall_only_vad_frms = self.get_acc(tgt_sig_vad, est_f_doa, ref_f_doa, tol=self.tol, vad_th=0.6)
			est_2mic_frm_Acc, est_2mic_mae_only_correct_frms, est_2mic_mae_only_incorrect_frms, est_2mic_mae_overall_only_vad_frms = self.get_acc(tgt_sig_vad, est_2mic_f_doa, ref_2mic_f_doa, tol=self.tol, vad_th=0.6)

			"""
			blk_size = 25
			mix_blk_vals = block_doa(frm_val=mix_f_vals, block_size=blk_size)
			tgt_blk_vals = block_doa(frm_val=tgt_f_vals, block_size=blk_size)
			est_blk_vals = block_doa(frm_val=est_f_vals, block_size=blk_size)

			#lbl_blk_doa, lbl_blk_range = block_lbl_doa(lbl_doa, block_size=blk_size)
			tgt_blk_vad = blk_vad(tgt_sig_vad, blk_size)
			#breakpoint()
			if "real_rirs" not in self.array_config:
				ref_blk_vals = tgt_blk_vals
			else:
				ref_blk_vals, _ = block_lbl_doa(doa, block_size=blk_size)#block_doa(frm_val=ref_f_doa, block_size=blk_size)

			mix_Acc = self.get_acc(tgt_blk_vad, mix_blk_vals, ref_blk_vals, tol=self.tol)
			est_Acc = self.get_acc(tgt_blk_vad, est_blk_vals, ref_blk_vals, tol=self.tol)
			"""
			#Utterance level
			
			est_utt_Acc = 1.0 if torch.abs(est_utt_doa-doa_degrees) <= self.tol else 0 #tgt_utt_doa

			est_2mic_utt_Acc = 1.0 if torch.abs(est_2mic_utt_doa-doa_degrees) <= self.tol else 0 #tgt_utt_doa
			
			
			self.log(f"{loss_flag}_est_frm_Acc", est_frm_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
			self.log(f"{loss_flag}_est_utt_Acc", est_utt_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

			#self.log("mix_blk_Acc", mix_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
			#self.log("est_blk_Acc", est_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
			
			self.log(f"{loss_flag}_est_2mic_frm_Acc", est_2mic_frm_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)			
			self.log(f"{loss_flag}_est_2mic_utt_Acc", est_2mic_utt_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

			#print(batch_idx, mix_frm_Acc, est_frm_Acc, mix_Acc, est_Acc)
			print(batch_idx, idx, loss_flag, mix_frm_Acc, est_frm_Acc, doa_degrees, mix_utt_doa, est_utt_doa, mix_utt_Acc, est_utt_Acc)
			print(batch_idx, idx, loss_flag, mix_2mic_frm_Acc, est_2mic_frm_Acc, doa_degrees, mix_2mic_utt_doa, est_2mic_utt_doa, mix_2mic_utt_Acc, est_2mic_utt_Acc)
			print('est', est_mae_only_correct_frms, est_mae_only_incorrect_frms, est_mae_overall_only_vad_frms, 'est_2mic', est_2mic_mae_only_correct_frms, est_2mic_mae_only_incorrect_frms, est_2mic_mae_overall_only_vad_frms)
			
			self.log(f"{loss_flag}_est_mae_only_correct_frms", est_mae_only_correct_frms, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
			self.log(f"{loss_flag}_est_mae_only_incorrect_frms", est_mae_only_incorrect_frms, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
			self.log(f"{loss_flag}_est_mae_overall_only_vad_frms", est_mae_overall_only_vad_frms, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

			#don't care for simulated rirs
			if "real_rirs" in self.array_config and ( not (doa_degrees==0 or doa_degrees==180)):

				#self.acc += est_utt_Acc
				#self.files += 1
				#print(batch_idx, self.acc, self.files)

				self.log("mix_frm_Acc_excluded_endfire", mix_frm_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
				self.log("est_frm_Acc_excluded_endfire", est_frm_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
				#self.log("mix_blk_Acc_excluded_endfire", mix_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
				#self.log("est_blk_Acc_excluded_endfire", est_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
				self.log("mix_utt_Acc_excluded_endfire", mix_utt_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
				self.log("est_utt_Acc_excluded_endfire", est_utt_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

				self.log("mix_2mic_frm_Acc_excluded_endfire", mix_2mic_frm_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
				self.log("est_2mic_frm_Acc_excluded_endfire", est_2mic_frm_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
				self.log("mix_2mic_utt_Acc_excluded_endfire", mix_2mic_utt_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
				self.log("est_2mic_utt_Acc_excluded_endfire", est_2mic_utt_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
				
			if self.dbg_log:
				torch.save({'mix': (mix_f_doa, mix_f_vals, mix_utt_doa),
							'tgt': (tgt_f_doa, tgt_f_vals, tgt_sig_vad, tgt_utt_doa),
							'est': (est_f_doa, est_f_vals, est_utt_doa),
							'doa': doa }, 
							f'../signals/simu_rirs_dbg/loss_functions_comparison_2mic_reverb/doa_{1.0}_{batch_idx}_{idx}_{loss_flag}_tol_{self.tol}deg_euclid_{self.euclid_dist}_wgt_mech_{self.wgt_mech}.pt', 
					)
			
		return
	
	def get_log_dir(self):
		log_base_dir = '/fs/scratch/PAS0774/Shanmukh/ControlledExp/random_seg/'
		log_dir_task = f'{log_base_dir}/dbg_signals/simu_rirs_dbg_analysis/loss_functions_comparison_num_mics/{self.dataset_condition}/'
		if "noisy" in self.dataset_condition:
			log_dir = f'{log_dir_task}{self.noise_simulation}/'
		else:
			log_dir = log_dir_task

		_log_dir = f'{log_dir}{self.log_str}_corrected_doa_sig_norm/'

		if not os.path.exists(_log_dir):
			os.makedirs(_log_dir)

		return _log_dir


	# for multiple loss functions testing
	def _batch_metrics_num_mics(self, batch, outputs, batch_idx):
		
		log_dir = self.get_log_dir()
		mix_ri_spec, tgt_ri_spec, doa = batch

		mix_sig = self.format_istft(mix_ri_spec)
		tgt_sig = self.format_istft(tgt_ri_spec)

		#normalization creates incorrect snr bcz of incorrect noise 
		#normalize the signals --- observed improves webrtc vad 
		#mix_sig = mix_sig/torch.max(torch.abs(mix_sig))
		#tgt_sig = tgt_sig/torch.max(torch.abs(tgt_sig))

		if self.dbg_log:
			torchaudio.save(f'{log_dir}mix_{batch_idx}.wav', mix_sig.cpu(), sample_rate=16000)  #/torch.max(torch.abs(mix_sig))
			torchaudio.save(f'{log_dir}tgt_{batch_idx}.wav', tgt_sig.cpu(), sample_rate=16000)  #/torch.max(torch.abs(tgt_sig)))
			
		mix_metrics_list = []
		for _ch in range(tgt_sig.shape[0]):
			mix_metrics = eval_metrics_v1(tgt_sig[_ch,:].cpu().numpy(), mix_sig[_ch,:].cpu().numpy())
			mix_metrics_list.append(mix_metrics)

		mix_metrics = eval_metrics_batch_v1(tgt_sig.cpu().numpy(), mix_sig.cpu().numpy())
		self.log("MIX_SNR", mix_metrics['snr'], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		self.log("MIX_STOI", mix_metrics['stoi'], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		self.log("MIX_ESTOI", mix_metrics['e_stoi'], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		self.log("MIX_PESQ_NB", mix_metrics['pesq_nb'], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		self.log("MIX_PESQ_WB", mix_metrics['pesq_wb'], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)


		#Vad 
		vad_comp_sig = tgt_sig/torch.max(torch.abs(tgt_sig)) if "real_rirs" not in self.array_config else est_sig

		num_frms =  (vad_comp_sig.shape[1]-self.frame_size +1)//self.frame_shift + 1
		
		tgt_sig_vad = torch.ones(num_frms)
		for idx in range(self.num_mics):
			sig_vad = compute_vad(vad_comp_sig[idx,:].cpu().numpy(), self.frame_size, self.frame_shift)
			tgt_sig_vad *= sig_vad 
			
		tgt_sig_vad = tgt_sig_vad.to(device=tgt_sig.device)


		mix_spec_cmplx = self.format_complex(mix_ri_spec)
		tgt_spec_cmplx = self.format_complex(tgt_ri_spec)

		if self.wgt_mech=="MASK":
			#mask
			mix_wt = self.get_mask(mix_spec_cmplx, mix_spec_cmplx)
			tgt_wt = self.get_mask(mix_spec_cmplx, tgt_spec_cmplx)
		else:
			mix_wt = torch.abs(mix_spec_cmplx)
			tgt_wt = torch.abs(tgt_spec_cmplx)
			

		## Computing mix, tgt frame level doa for (8, 4, 2) mics
		
		mix_8mic_f_doa, mix_f_vals, mix_8mic_utt_doa, mix_2mic_f_doa, mix_2mic_utt_doa, mix_2mic_f_vals = gcc_phat_all_pairs(mix_spec_cmplx, mix_wt, 16000, self.frame_size, self.local_mic_pos, 
								 										self.local_mic_center, src_mic_dist=1, weighted=True, sig_vad=tgt_sig_vad, is_euclidean_dist=self.euclid_dist, mic_pairs=self.mic_pairs)
		tgt_8mic_f_doa, tgt_f_vals, tgt_8mic_utt_doa, tgt_2mic_f_doa, tgt_2mic_utt_doa, tgt_2mic_f_vals = gcc_phat_all_pairs(tgt_spec_cmplx, tgt_wt, 16000, self.frame_size, self.local_mic_pos, 
								 										self.local_mic_center, src_mic_dist=1, weighted=True, sig_vad=tgt_sig_vad, is_euclidean_dist=self.euclid_dist, mic_pairs=self.mic_pairs)
		
		centre_mic_idx = self.local_mic_pos.shape[0]//2

		mic_pos_4 = self.local_mic_pos[centre_mic_idx-2:centre_mic_idx+2,:]  
		mic_pairs_4 = [(mic_1, mic_2) for mic_1 in range(0, 4) for mic_2 in range(mic_1+1, 4)]

		mix_spec_cmplx_4mic = mix_spec_cmplx[centre_mic_idx-2:centre_mic_idx+2,:,:]
		mix_wt_4mic = mix_wt[centre_mic_idx-2:centre_mic_idx+2,:,:]

		tgt_spec_cmplx_4mic = tgt_spec_cmplx[centre_mic_idx-2:centre_mic_idx+2,:,:]
		tgt_wt_4mic = tgt_wt[centre_mic_idx-2:centre_mic_idx+2,:,:]
		
		mix_4mic_f_doa, mix_f_vals, mix_4mic_utt_doa, mix_2mic_f_doa, mix_2mic_utt_doa, mix_2mic_f_vals = gcc_phat_all_pairs(mix_spec_cmplx_4mic, mix_wt_4mic, 16000, self.frame_size, mic_pos_4, 
								 										self.local_mic_center, src_mic_dist=1, weighted=True, sig_vad=tgt_sig_vad, is_euclidean_dist=self.euclid_dist, mic_pairs=mic_pairs_4)
		tgt_4mic_f_doa, tgt_f_vals, tgt_4mic_utt_doa, tgt_2mic_f_doa, tgt_2mic_utt_doa, tgt_2mic_f_vals = gcc_phat_all_pairs(tgt_spec_cmplx_4mic, tgt_wt_4mic, 16000, self.frame_size, mic_pos_4, 
								 										self.local_mic_center, src_mic_dist=1, weighted=True, sig_vad=tgt_sig_vad, is_euclidean_dist=self.euclid_dist, mic_pairs=mic_pairs_4)
		
		if "real_rirs" not in self.array_config:
			ref_8mic_f_doa = tgt_8mic_f_doa
			ref_2mic_f_doa = tgt_2mic_f_doa
			ref_4mic_f_doa = tgt_4mic_f_doa
		else:
			ref_f_doa = torch.rad2deg(doa[:,:,-1])[0,:mix_2mic_f_doa.shape[0]]
			ref_2mic_f_doa = ref_f_doa
		
		mix_8mic_frm_Acc, mix_8mic_mae_only_correct_frms, mix_8mic_mae_only_incorrect_frms, mix_8mic_mae_overall_only_vad_frms = self.get_acc(tgt_sig_vad, mix_8mic_f_doa, ref_8mic_f_doa, tol=self.tol, vad_th=0.6)
		mix_2mic_frm_Acc, mix_2mic_mae_only_correct_frms, mix_2mic_mae_only_incorrect_frms, mix_2mic_mae_overall_only_vad_frms = self.get_acc(tgt_sig_vad, mix_2mic_f_doa, ref_2mic_f_doa, tol=self.tol, vad_th=0.6)
		mix_4mic_frm_Acc, mix_4mic_mae_only_correct_frms, mix_4mic_mae_only_incorrect_frms, mix_4mic_mae_overall_only_vad_frms = self.get_acc(tgt_sig_vad, mix_4mic_f_doa, ref_4mic_f_doa, tol=self.tol, vad_th=0.6)

		print('mix_8mic_mae_info', mix_8mic_mae_only_correct_frms, mix_8mic_mae_only_incorrect_frms, mix_8mic_mae_overall_only_vad_frms)
		print('mix_4mic_mae_info', mix_4mic_mae_only_correct_frms, mix_4mic_mae_only_incorrect_frms, mix_4mic_mae_overall_only_vad_frms)
		print('mix_2mic_mae_info', mix_2mic_mae_only_correct_frms, mix_2mic_mae_only_incorrect_frms, mix_2mic_mae_overall_only_vad_frms)

		doa_degrees = torch.abs(torch.rad2deg(doa[:,:,-1])[0,0])
		mix_8mic_utt_Acc = 1.0 if torch.abs(mix_8mic_utt_doa-doa_degrees) <= self.tol else 0 #tgt_utt_doa
		mix_4mic_utt_Acc = 1.0 if torch.abs(mix_4mic_utt_doa-doa_degrees) <= self.tol else 0 #tgt_utt_doa
		mix_2mic_utt_Acc = 1.0 if torch.abs(mix_2mic_utt_doa-doa_degrees) <= self.tol else 0 #tgt_utt_doa

		self.log(f"mix_8mic_frm_Acc", mix_8mic_frm_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		self.log(f"mix_4mic_frm_Acc", mix_4mic_frm_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		self.log(f"mix_2mic_frm_Acc", mix_2mic_frm_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

		self.log(f"mix_8mic_utt_Acc", mix_8mic_utt_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		self.log(f"mix_4mic_utt_Acc", mix_4mic_utt_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		self.log(f"mix_2mic_utt_Acc", mix_2mic_utt_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		
		self.log(f"mix_8mic_mae_only_correct_frms", mix_8mic_mae_only_correct_frms, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		self.log(f"mix_8mic_mae_only_incorrect_frms", mix_8mic_mae_only_incorrect_frms, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		self.log(f"mix_8mic_mae_overall_only_vad_frms", mix_8mic_mae_overall_only_vad_frms, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

		self.log(f"mix_2mic_mae_only_correct_frms", mix_2mic_mae_only_correct_frms, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		self.log(f"mix_2mic_mae_only_incorrect_frms", mix_2mic_mae_only_incorrect_frms, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		self.log(f"mix_2mic_mae_overall_only_vad_frms", mix_2mic_mae_overall_only_vad_frms, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

		self.log(f"mix_4mic_mae_only_correct_frms", mix_4mic_mae_only_correct_frms, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		self.log(f"mix_4mic_mae_only_incorrect_frms", mix_4mic_mae_only_incorrect_frms, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		self.log(f"mix_4mic_mae_overall_only_vad_frms", mix_4mic_mae_overall_only_vad_frms, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		est_dict = {}
		
		for key_num_mic_str in ['est_2mic_ri_spec', 'est_4mic_ri_spec', 'est_8mic_ri_spec']:
	
			if '2mic' in key_num_mic_str:	
				mic_pos = self.local_mic_pos[centre_mic_idx-1:centre_mic_idx+1,:]  
				mic_pairs = [(0,1)]

				mix_spec_cmplx_mics = mix_spec_cmplx[centre_mic_idx-1:centre_mic_idx+1,:,:]

				ref_f_doa = ref_2mic_f_doa
				mix_frm_Acc, mix_utt_doa, mix_utt_Acc = mix_2mic_frm_Acc, mix_2mic_utt_doa, mix_2mic_utt_Acc

				#loss_info
				loss_2mic_info = outputs['loss_2mic_info']

			elif '4mic' in key_num_mic_str:	
				mic_pos = self.local_mic_pos[centre_mic_idx-2:centre_mic_idx+2,:]  
				mic_pairs = [(mic_1, mic_2) for mic_1 in range(0, 4) for mic_2 in range(mic_1+1, 4)]

				mix_spec_cmplx_mics = mix_spec_cmplx[centre_mic_idx-2:centre_mic_idx+2,:,:]
				ref_f_doa = ref_4mic_f_doa
				mix_frm_Acc, mix_utt_doa, mix_utt_Acc = mix_4mic_frm_Acc, mix_4mic_utt_doa, mix_4mic_utt_Acc

				loss_4mic_info = outputs['loss_4mic_info']
			else:
				mic_pos = self.local_mic_pos
				mic_pairs = self.mic_pairs

				mix_spec_cmplx_mics = mix_spec_cmplx
				ref_f_doa = ref_8mic_f_doa
				mix_frm_Acc, mix_utt_doa, mix_utt_Acc = mix_8mic_frm_Acc, mix_8mic_utt_doa, mix_8mic_utt_Acc

				loss_8mic_info = outputs['loss_8mic_info']

			est_ri_spec_all = outputs[key_num_mic_str]  #(b, n_ch, T, F)

			num_batches = est_ri_spec_all.shape[0]
			num_mics = est_ri_spec_all.shape[1]//2

			## SE metrics
			#adjusting shape, type for istft
			
			est_sig_all = self.format_istft(est_ri_spec_all)
			

			est_sig_all = est_sig_all #/torch.max(torch.abs(est_sig_all))

			# loss functions
			for idx in range(num_batches):
				loss_flag = self.loss_flags[idx]

				est_sig = est_sig_all[idx*num_mics:idx*num_mics+num_mics,:]
				est_sig = est_sig #/torch.max(torch.abs(est_sig))
				if self.dbg_log:
					torchaudio.save(f'{log_dir}{key_num_mic_str[:8]}_{batch_idx}_{idx}_{loss_flag}.wav', est_sig.cpu(), sample_rate=16000) #/torch.max(torch.abs(est_sig)))

				metrics_list = [] # list of dictionaries
				for _ch in range(num_mics):
					_metrics = eval_metrics_v1(tgt_sig[_ch,:].cpu().numpy(), est_sig[_ch,:].cpu().numpy())
					self.log(f"{loss_flag}_SNR_{key_num_mic_str[:8]}_{_ch}", _metrics['snr'], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
					self.log(f"{loss_flag}_STOI_{key_num_mic_str[:8]}_{_ch}", _metrics['stoi'], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
					self.log(f"{loss_flag}_ESTOI_{key_num_mic_str[:8]}_{_ch}", _metrics['e_stoi'], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
					self.log(f"{loss_flag}_PESQ_NB_{key_num_mic_str[:8]}_{_ch}", _metrics['pesq_nb'], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
					self.log(f"{loss_flag}_PESQ_WB_{key_num_mic_str[:8]}_{_ch}", _metrics['pesq_wb'], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

					metrics_list.append(_metrics)


				est_ri_spec = est_ri_spec_all[[idx]]
				est_spec_cmplx = self.format_complex(est_ri_spec)

				#removing spatial aliasing freq bins
				#print(mix_spec_cmplx.shape, tgt_spec_cmplx.shape, est_spec_cmplx.shape)
				if self.wgt_mech=="MASK":
				#mask
					est_wt = self.get_mask(mix_spec_cmplx_mics, est_spec_cmplx)
				else:
					est_wt = torch.abs(est_spec_cmplx)
			
				est_f_doa, est_f_vals, est_utt_doa, est_2mic_f_doa, est_2mic_utt_doa, est_2mic_f_vals = gcc_phat_all_pairs(est_spec_cmplx, est_wt, 16000, self.frame_size, mic_pos, 
								 										self.local_mic_center, src_mic_dist=1, weighted=True, sig_vad=tgt_sig_vad, is_euclidean_dist=self.euclid_dist, mic_pairs=mic_pairs)


				
				est_frm_Acc, est_mae_only_correct_frms, est_mae_only_incorrect_frms, est_mae_overall_only_vad_frms = self.get_acc(tgt_sig_vad, est_f_doa, ref_f_doa, tol=self.tol, vad_th=0.6)
				est_2mic_frm_Acc, est_2mic_mae_only_correct_frms, est_2mic_mae_only_incorrect_frms, est_2mic_mae_overall_only_vad_frms = self.get_acc(tgt_sig_vad, est_2mic_f_doa, ref_2mic_f_doa, tol=self.tol, vad_th=0.6)

				if '8mic' in key_num_mic_str:
					est_4mic_f_doa, est_4mic_f_vals, est_4mic_utt_doa, est_2mic_f_doa, est_2mic_utt_doa, est_2mic_f_vals = gcc_phat_all_pairs(est_spec_cmplx[2:6,:,:], est_wt[2:6,:,:], 16000, self.frame_size, mic_pos[2:6,:], 
								 										self.local_mic_center, src_mic_dist=1, weighted=True, sig_vad=tgt_sig_vad, is_euclidean_dist=self.euclid_dist, mic_pairs=mic_pairs_4)
					est_4mic_frm_Acc, est_4mic_mae_only_correct_frms, est_4mic_mae_only_incorrect_frms, est_4mic_mae_overall_only_vad_frms = self.get_acc(tgt_sig_vad, est_4mic_f_doa, ref_4mic_f_doa, tol=self.tol, vad_th=0.6)

					est_4mic_utt_Acc = 1.0 if torch.abs(est_4mic_utt_doa-doa_degrees) <= self.tol else 0 #tgt_utt_doa
				"""
				blk_size = 25
				mix_blk_vals = block_doa(frm_val=mix_f_vals, block_size=blk_size)
				tgt_blk_vals = block_doa(frm_val=tgt_f_vals, block_size=blk_size)
				est_blk_vals = block_doa(frm_val=est_f_vals, block_size=blk_size)

				#lbl_blk_doa, lbl_blk_range = block_lbl_doa(lbl_doa, block_size=blk_size)
				tgt_blk_vad = blk_vad(tgt_sig_vad, blk_size)
				#breakpoint()
				if "real_rirs" not in self.array_config:
					ref_blk_vals = tgt_blk_vals
				else:
					ref_blk_vals, _ = block_lbl_doa(doa, block_size=blk_size)#block_doa(frm_val=ref_f_doa, block_size=blk_size)

				mix_Acc = self.get_acc(tgt_blk_vad, mix_blk_vals, ref_blk_vals, tol=self.tol)
				est_Acc = self.get_acc(tgt_blk_vad, est_blk_vals, ref_blk_vals, tol=self.tol)
				"""
				#Utterance level
				
				est_utt_Acc = 1.0 if torch.abs(est_utt_doa-doa_degrees) <= self.tol else 0 #tgt_utt_doa
				est_2mic_utt_Acc = 1.0 if torch.abs(est_2mic_utt_doa-doa_degrees) <= self.tol else 0 #tgt_utt_doa
			
			
				self.log(f"{loss_flag}_{key_num_mic_str[:8]}_enh_frm_Acc", est_frm_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
				self.log(f"{loss_flag}_{key_num_mic_str[:8]}_enh_utt_Acc", est_utt_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

				self.log(f"{loss_flag}_{key_num_mic_str[:8]}_enh_est_mae_only_correct_frms", est_mae_only_correct_frms, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
				self.log(f"{loss_flag}_{key_num_mic_str[:8]}_enh_est_mae_only_incorrect_frms", est_mae_only_incorrect_frms, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
				self.log(f"{loss_flag}_{key_num_mic_str[:8]}_enh_est_mae_overall_only_vad_frms", est_mae_overall_only_vad_frms, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

				print(f'{key_num_mic_str[:8]}', batch_idx, idx, loss_flag, mix_frm_Acc, est_frm_Acc, doa_degrees, mix_utt_doa, est_utt_doa, mix_utt_Acc, est_utt_Acc)

				#self.log("mix_blk_Acc", mix_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
				#self.log("est_blk_Acc", est_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
				if num_mics>4:
					self.log(f"{loss_flag}_{key_num_mic_str[:8]}_enh_4mic_frm_Acc", est_4mic_frm_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)			
					self.log(f"{loss_flag}_{key_num_mic_str[:8]}_enh_4mic_utt_Acc", est_4mic_utt_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

					self.log(f"{loss_flag}_{key_num_mic_str[:8]}_enh_2mic_frm_Acc", est_2mic_frm_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)			
					self.log(f"{loss_flag}_{key_num_mic_str[:8]}_enh_2mic_utt_Acc", est_2mic_utt_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

					self.log(f"{loss_flag}_{key_num_mic_str[:8]}_enh_est_4mic_mae_only_correct_frms", est_4mic_mae_only_correct_frms, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
					self.log(f"{loss_flag}_{key_num_mic_str[:8]}_enh_est_4mic_mae_only_incorrect_frms", est_4mic_mae_only_incorrect_frms, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
					self.log(f"{loss_flag}_{key_num_mic_str[:8]}_enh_est_4mic_mae_overall_only_vad_frms", est_4mic_mae_overall_only_vad_frms, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

					self.log(f"{loss_flag}_{key_num_mic_str[:8]}_enh_est_2mic_mae_only_correct_frms", est_2mic_mae_only_correct_frms, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
					self.log(f"{loss_flag}_{key_num_mic_str[:8]}_enh_est_2mic_mae_only_incorrect_frms", est_2mic_mae_only_incorrect_frms, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
					self.log(f"{loss_flag}_{key_num_mic_str[:8]}_enh_est_2mic_mae_overall_only_vad_frms", est_2mic_mae_overall_only_vad_frms, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

					print(f'{key_num_mic_str[:8]}_4mic', batch_idx, idx, loss_flag, mix_4mic_frm_Acc, est_4mic_frm_Acc, doa_degrees, mix_4mic_utt_doa, est_4mic_utt_doa, mix_4mic_utt_Acc, est_4mic_utt_Acc)
					print(f'{key_num_mic_str[:8]}_2mic', batch_idx, idx, loss_flag, mix_2mic_frm_Acc, est_2mic_frm_Acc, doa_degrees, mix_2mic_utt_doa, est_2mic_utt_doa, mix_2mic_utt_Acc, est_2mic_utt_Acc)

				elif num_mics>2:
					self.log(f"{loss_flag}_{key_num_mic_str[:8]}_enh_2mic_frm_Acc", est_2mic_frm_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)			
					self.log(f"{loss_flag}_{key_num_mic_str[:8]}_enh_2mic_utt_Acc", est_2mic_utt_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

					self.log(f"{loss_flag}_{key_num_mic_str[:8]}_enh_est_2mic_mae_only_correct_frms", est_2mic_mae_only_correct_frms, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
					self.log(f"{loss_flag}_{key_num_mic_str[:8]}_enh_est_2mic_mae_only_incorrect_frms", est_2mic_mae_only_incorrect_frms, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
					self.log(f"{loss_flag}_{key_num_mic_str[:8]}_enh_est_2mic_mae_overall_only_vad_frms", est_2mic_mae_overall_only_vad_frms, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

					print(f'{key_num_mic_str[:8]}_2mic', batch_idx, idx, loss_flag, mix_2mic_frm_Acc, est_2mic_frm_Acc, doa_degrees, mix_2mic_utt_doa, est_2mic_utt_doa, mix_2mic_utt_Acc, est_2mic_utt_Acc)

				#print(batch_idx, mix_frm_Acc, est_frm_Acc, mix_Acc, est_Acc)
				
				
				print('est', est_mae_only_correct_frms, est_mae_only_incorrect_frms, est_mae_overall_only_vad_frms, 'est_2mic', est_2mic_mae_only_correct_frms, est_2mic_mae_only_incorrect_frms, est_2mic_mae_overall_only_vad_frms)
				
	
				#don't care for simulated rirs
				if "real_rirs" in self.array_config and ( not (doa_degrees==0 or doa_degrees==180)):

					#self.acc += est_utt_Acc
					#self.files += 1
					#print(batch_idx, self.acc, self.files)

					self.log("mix_frm_Acc_excluded_endfire", mix_frm_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
					self.log("est_frm_Acc_excluded_endfire", est_frm_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
					#self.log("mix_blk_Acc_excluded_endfire", mix_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
					#self.log("est_blk_Acc_excluded_endfire", est_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
					self.log("mix_utt_Acc_excluded_endfire", mix_utt_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
					self.log("est_utt_Acc_excluded_endfire", est_utt_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

					self.log("mix_2mic_frm_Acc_excluded_endfire", mix_2mic_frm_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
					self.log("est_2mic_frm_Acc_excluded_endfire", est_2mic_frm_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
					self.log("mix_2mic_utt_Acc_excluded_endfire", mix_2mic_utt_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
					self.log("est_2mic_utt_Acc_excluded_endfire", est_2mic_utt_Acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
				

				est_dict[f'{loss_flag}_{key_num_mic_str[:8]}_enh'] = (est_f_doa, est_f_vals, est_utt_doa, est_frm_Acc, metrics_list, est_mae_only_correct_frms, est_mae_only_incorrect_frms, est_mae_overall_only_vad_frms)
				if num_mics>4:
					est_dict[f'{loss_flag}_{key_num_mic_str[:8]}_enh_4mic'] = (est_4mic_f_doa, est_4mic_f_vals, est_4mic_utt_doa, est_4mic_frm_Acc, est_4mic_mae_only_correct_frms, est_4mic_mae_only_incorrect_frms, est_4mic_mae_overall_only_vad_frms)
					est_dict[f'{loss_flag}_{key_num_mic_str[:8]}_enh_2mic'] = (est_2mic_f_doa, est_2mic_f_vals, est_2mic_utt_doa, est_2mic_frm_Acc, est_2mic_mae_only_correct_frms, est_2mic_mae_only_incorrect_frms, est_2mic_mae_overall_only_vad_frms)
				elif num_mics>2:
					est_dict[f'{loss_flag}_{key_num_mic_str[:8]}_enh_2mic'] = (est_2mic_f_doa, est_2mic_f_vals, est_2mic_utt_doa, est_2mic_frm_Acc, est_2mic_mae_only_correct_frms, est_2mic_mae_only_incorrect_frms, est_2mic_mae_overall_only_vad_frms)
		
		self._print_doa_num_mic_var(batch_idx, num_batches, est_dict, mix_2mic_frm_Acc, mix_4mic_frm_Acc, mix_8mic_frm_Acc, loss_2mic_info, loss_4mic_info, loss_8mic_info)
		self._print_metrics_num_mic_var(batch_idx, num_batches, est_dict, mix_metrics_list)
		
		if self.dbg_log:
			torch.save({'mix': (mix_f_doa, mix_f_vals, mix_utt_doa),
						'tgt': (tgt_f_doa, tgt_f_vals, tgt_sig_vad, tgt_utt_doa),
						'ests': est_dict, #(est_f_doa, est_f_vals, est_utt_doa),
						'doa': doa }, 
						f'{log_dir}doa_{batch_idx}.pt', 
				)
			
		return


	def _print_doa_num_mic_var(self, batch_idx, num_batches, exp_dict, mix_2mic_frm_Acc, mix_4mic_frm_Acc, mix_8mic_frm_Acc, loss_2mic_info, loss_4mic_info, loss_8mic_info):
		#print(f'{"Frm Acc":45}', '2mic', '4mic', '8mic')
		file_name = f'../Logs/{self.dataset_condition}_8cm_{self.log_str}_doa.csv'
		f = open(file_name,'a')
		doa_writer = csv.writer(f)
		mix_str = [f'{batch_idx:04d}', f'{"Unproc":25}', 'mix_nmic_ri_spec',  f'{mix_2mic_frm_Acc:.4f}', f'{mix_4mic_frm_Acc:.4f}', f'{mix_8mic_frm_Acc:.4f}']
		doa_writer.writerow(mix_str)
		for idx in range(num_batches):
			loss_flag = self.loss_flags[idx]
			for key_num_mic_str in ['est_2mic_ri_spec', 'est_4mic_ri_spec', 'est_8mic_ri_spec']:
				matched_mic_str = f'{loss_flag}_{key_num_mic_str[:8]}_enh'
				if matched_mic_str in exp_dict.keys():
					(est_f_doa, est_f_vals, est_utt_doa, est_frm_Acc, metrics_list, est_mae_only_correct_frms, est_mae_only_incorrect_frms, est_mae_overall_only_vad_frms) = exp_dict[matched_mic_str]
					if '2mic' in key_num_mic_str:
						loss_ri, loss_mag, loss_phdiff = loss_2mic_info[idx]
						_str = [f'{batch_idx:04d}', f'{loss_flag:25}', key_num_mic_str, f'{est_frm_Acc:.4f}', 'xxxxxx', 'xxxxxx', f'{loss_ri:.4f}', f'{loss_mag:.4f}', f'{loss_phdiff:.4f}', f'{est_mae_only_correct_frms:.2f}', f'{est_mae_only_incorrect_frms:.2f}', f'{est_mae_overall_only_vad_frms:.2f}', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx', 'xxxxx',]
						
						
					elif '4mic' in key_num_mic_str:
						matched_2mic_mic_str = f'{loss_flag}_{key_num_mic_str[:8]}_enh_2mic'
						(est_2mic_f_doa, est_2mic_f_vals, est_2mic_utt_doa, est_2mic_frm_Acc, est_2mic_mae_only_correct_frms, est_2mic_mae_only_incorrect_frms, est_2mic_mae_overall_only_vad_frms) = exp_dict[matched_2mic_mic_str]

						loss_ri, loss_mag, loss_phdiff = loss_4mic_info[idx]

						_str = [f'{batch_idx:04d}', f'{loss_flag:25}', key_num_mic_str, f'{est_2mic_frm_Acc:.4f}', f'{est_frm_Acc:.4f}', 'xxxxxx', f'{loss_ri:.4f}', f'{loss_mag:.4f}', f'{loss_phdiff:.4f}', f'{est_2mic_mae_only_correct_frms:.2f}', f'{est_2mic_mae_only_incorrect_frms:.2f}', f'{est_2mic_mae_overall_only_vad_frms:.2f}', f'{est_mae_only_correct_frms:.2f}', f'{est_mae_only_incorrect_frms:.2f}', f'{est_mae_overall_only_vad_frms:.2f}', 'xxxxx', 'xxxxx', 'xxxxx' ]
					elif '8mic' in key_num_mic_str:
						matched_2mic_mic_str = f'{loss_flag}_{key_num_mic_str[:8]}_enh_2mic'
						matched_4mic_mic_str = f'{loss_flag}_{key_num_mic_str[:8]}_enh_4mic'

						(est_2mic_f_doa, est_2mic_f_vals, est_2mic_utt_doa, est_2mic_frm_Acc, est_2mic_mae_only_correct_frms, est_2mic_mae_only_incorrect_frms, est_2mic_mae_overall_only_vad_frms) = exp_dict[matched_2mic_mic_str]
						(est_4mic_f_doa, est_4mic_f_vals, est_4mic_utt_doa, est_4mic_frm_Acc, est_4mic_mae_only_correct_frms, est_4mic_mae_only_incorrect_frms, est_4mic_mae_overall_only_vad_frms) = exp_dict[matched_4mic_mic_str]

						loss_ri, loss_mag, loss_phdiff = loss_8mic_info[idx]

						_str = [f'{batch_idx:04d}', f'{loss_flag:25}', key_num_mic_str, f'{est_2mic_frm_Acc:.4f}', f'{est_4mic_frm_Acc:.4f}', f'{est_frm_Acc:.4f}', f'{loss_ri:.4f}', f'{loss_mag:.4f}', f'{loss_phdiff:.4f}', f'{est_2mic_mae_only_correct_frms:.2f}', f'{est_2mic_mae_only_incorrect_frms:.2f}', f'{est_2mic_mae_overall_only_vad_frms:.2f}', f'{est_4mic_mae_only_correct_frms:.2f}', f'{est_4mic_mae_only_incorrect_frms:.2f}', f'{est_4mic_mae_overall_only_vad_frms:.2f}', f'{est_mae_only_correct_frms:.2f}', f'{est_mae_only_incorrect_frms:.2f}', f'{est_mae_overall_only_vad_frms:.2f}']
				
					doa_writer.writerow(_str)
		
		f.close()


	def get_metrics_per_channel(self, batch_idx, metrics_list, metric, loss_flag):
		num_mics = len(metrics_list)
	
		ch_metric = []
		for _mic in range(num_mics):
			val = metrics_list[_mic][metric]
			if 'stoi' in metric:
				val = round(val*100,2)
			elif 'pesq' in metric or 'snr' in metric:
				val = round(val,2)

			ch_metric.append(val)
		
		
		if 2==num_mics:
			_str = [f'{batch_idx:04d}', f'{loss_flag:25}', f'{num_mics}_{metric:10}', 'xxxxx', 'xxxxx', 'xxxxx', f'{ch_metric[0]:.2f}', f'{ch_metric[1]:.2f}', 'xxxxx', 'xxxxx', 'xxxxx']
		elif 4==num_mics:
			_str = [f'{batch_idx:04d}', f'{loss_flag:25}', f'{num_mics}_{metric:10}', 'xxxxx', 'xxxxx', f'{ch_metric[0]:.2f}', f'{ch_metric[1]:.2f}', f'{ch_metric[2]:.2f}', f'{ch_metric[3]:.2f}', 'xxxxx', 'xxxxx']
		elif 8==num_mics:
			_str = [f'{batch_idx:04d}', f'{loss_flag:25}', f'{num_mics}_{metric:10}', f'{ch_metric[0]:.2f}', f'{ch_metric[1]:.2f}', f'{ch_metric[2]:.2f}', f'{ch_metric[3]:.2f}', f'{ch_metric[4]:.2f}', f'{ch_metric[5]:.2f}', f'{ch_metric[6]:.2f}', f'{ch_metric[7]:.2f}']
		
		#print(_str)
		
		return _str


	def _print_metrics_num_mic_var(self, batch_idx, num_batches, exp_dict, mix_metrics_list):
		#print(f'{"Metrics":34}', '  0  ', '  1  ', '  2  ', '  3  ', '  4  ', '  5  ', '  6  ', '  7  ')
		file_name = f'../Logs/{self.dataset_condition}_8cm_{self.log_str}_metrics.csv'
		f = open(file_name,'a')
		metrics_writer = csv.writer(f)
		for metric in ['stoi', 'e_stoi', 'pesq_nb', 'snr']:
			mix_str = self.get_metrics_per_channel(batch_idx, mix_metrics_list, metric, 'Mix')
			metrics_writer.writerow(mix_str)
			for idx in range(num_batches):
				loss_flag = self.loss_flags[idx]
				for key_num_mic_str in ['est_2mic_ri_spec', 'est_4mic_ri_spec', 'est_8mic_ri_spec']:
					matched_mic_str = f'{loss_flag}_{key_num_mic_str[:8]}_enh'
					if matched_mic_str in exp_dict.keys():
						(est_f_doa, est_f_vals, est_utt_doa, est_frm_Acc, metrics_list, est_mae_only_correct_frms, est_mae_only_incorrect_frms, est_mae_overall_only_vad_frms) = exp_dict[matched_mic_str]
						
						_str = self.get_metrics_per_channel(batch_idx, metrics_list, metric, loss_flag)
						metrics_writer.writerow(_str)
		f.close()

		
	def on_test_batch_end(
		self,
		trainer: "pl.Trainer",
		pl_module: "pl.LightningModule",
		outputs: Optional[STEP_OUTPUT],
		batch: Any,
		batch_idx: int,
		dataloader_idx: int,
		) -> None:

		self._batch_metrics_v3(batch, outputs, batch_idx) #v3
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

		self._batch_metrics(batch, outputs, batch_idx)
