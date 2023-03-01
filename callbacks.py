import torch 
import torchaudio
import torchaudio.transforms as T
from torch.nn import functional as F
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT

from typing import Any, Optional

from metrics import eval_metrics_batch_v1, _mag_spec_mask, gettdsignal_mag_spec_mask
from masked_gcc_phat import gcc_phat, compute_vad, gcc_phat_loc_orient, block_doa, blk_vad, block_lbl_doa #, get_acc
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
	def __init__(self, array_config, dataset_dtype=None, dataset_condition=None, doa_tol = 5, doa_euclid_dist=False):
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
		self.tol = doa_tol
		self.euclid_dist = doa_euclid_dist
		#self.acc = 0
		#self.files = 0

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
		
		mix_frm_Acc = self.get_acc(tgt_sig_vad, mix_f_doa, ref_f_doa, tol=self.tol, vad_th=0.6)
		est_frm_Acc = self.get_acc(tgt_sig_vad, est_f_doa, ref_f_doa, tol=self.tol, vad_th=0.6)


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

	def on_test_batch_end(
		self,
		trainer: "pl.Trainer",
		pl_module: "pl.LightningModule",
		outputs: Optional[STEP_OUTPUT],
		batch: Any,
		batch_idx: int,
		dataloader_idx: int,
		) -> None:

		self._batch_metrics(batch, outputs, batch_idx)
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
