from train import DCCRN_model
from dataset import MovingSourceDataset, NetworkInput
from array_setup import array_setup_10cm_2mic
from masked_gcc_phat import gcc_phat
from metrics import eval_metrics_batch_v1

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


import pytorch_lightning as pl
from pytorch_lightning.profiler import SimpleProfiler
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, ModelSummary, LearningRateMonitor

import numpy as np
import torchaudio

class DOA_MISO_fwk(pl.LightningModule):
	def __init__(self, array_setup):
		super().__init__()
		self.array_setup = array_setup
		self.bidirectional = True
		
		self.model_path_0 = '/fs/scratch/PAS0774/Shanmukh/ControlledExp/MovingSrc/DCCRN/ref_mic_0/epoch=50-step=2295.ckpt'
		self.model_path_1 = '/fs/scratch/PAS0774/Shanmukh/ControlledExp/MovingSrc/DCCRN/ref_mic_1/epoch=50-step=2295.ckpt'
		self.model_0 = DCCRN_model.load_from_checkpoint(self.model_path_0, bidirectional=self.bidirectional, train_dataset=None, val_dataset=None)
		self.model_1 = DCCRN_model.load_from_checkpoint(self.model_path_1, bidirectional=self.bidirectional, train_dataset=None, val_dataset=None)

	
	def forward(self, input_batch):
		mix_ri_spec, tgt_ri_spec, doa = input_batch

		est_ri_spec_0, _ = self.model_0(input_batch)
		est_ri_spec_1, _ = self.model_1(input_batch)
		est_ri_spec = torch.concat((est_ri_spec_0, est_ri_spec_1), dim=1)

		return est_ri_spec

	def test_step(self, test_batch, batch_idx):
		est_ri_spec = self.forward(test_batch)
		return {"est_ri_spec" : est_ri_spec }


	def format_istft(self, mix_ri_spec):
		#adjusting shape, type for istft
		#(batch_size, 4, T, F) . -> (batch_size, F, T, 2)
		mix_ri_spec = mix_ri_spec.to(torch.float32)

		(batch_size, n_ch, n_frms, n_freq) = mix_ri_spec.shape

		intm_mix_ri_spec = torch.reshape(mix_ri_spec, (batch_size*n_ch, n_frms, n_freq))
		intm_mix_ri_spec = torch.reshape(intm_mix_ri_spec, (intm_mix_ri_spec.shape[0]//2, 2, n_frms, n_freq))
		_mix_ri_spec = torch.permute(intm_mix_ri_spec,[0,3,2,1])

		mix_sig = torch.istft(_mix_ri_spec, 320,160,320,torch.hamming_window(320).type_as(_mix_ri_spec), center=False)
		return mix_sig

if __name__=="__main__":

	if 0:
		static_prob = 0.0
		non_linear_motion_prob = 0.0
		T = 4
		nb_points = 16
		test_snr = 5
		test_t60 = 1.0
		ref_mic_idx = -1
		tst_mvng_src_dataset_config = MovingSourceDatasetConfig(T,fs=16000,array_setup=array_setup_10cm_2mic, dataset_type = "test",  
																static_prob=static_prob, non_linear_motion_prob=non_linear_motion_prob, 
																nb_points=nb_points, test_snr=test_snr, test_t60=test_t60, ref_mic_idx = ref_mic_idx,
																size=1)
		test_dataset = tst_mvng_src_dataset_config.dataset
		
		"""
		tb_logger = pl_loggers.TensorBoardLogger(save_dir=args.ckpt_dir, version=args.exp_name)

		trainer = pl.Trainer(accelerator='gpu', devices=1, num_nodes=1, precision=16,
							callbacks=[DOAcallbacks()]
							#logger=tb_logger
							)
		bidirectional = True #args.bidirectional
		"""
		"""
		msg = f"Test Config: bidirectional: {bidirectional}, T: {T}, \
			static_prob: {static_prob}, non_linear_motion_prob: {non_linear_motion_prob}, \
			nb_points: {nb_points}, batch_sie: {args.batch_size}, ckpt_dir: {args.ckpt_dir}, \
			model: {args.model_path}, ref_mic_idx : {ref_mic_idx}, snr: {args.test_snr}, test_t60: {args.test_t60} \n"
		print(msg)
		"""
		#trainer.logger.experiment.add_text("Exp details", msg)
	else:
		#Static Type
		ref_mic_idx = -1

		snr = -5
		t60 = 0.2
		src_mic_dist = 1.0
		noi_mic_dist = 1.0
		scenario = "source_moving"
		model_train = "moving"

		dataset_file = '../val_dataset_file_circular_motion_snr_-5_t60_0.2.txt'
		test_dataset = MovingSourceDataset(dataset_file, array_setup_10cm_2mic, transforms=[ NetworkInput(320, 160, ref_mic_idx)], size=5) #

		
	test_loader = DataLoader(test_dataset, batch_size = 1, num_workers=0, pin_memory=True, drop_last=True)

	model = DOA_MISO_fwk(array_setup = array_setup_10cm_2mic)
	#trainer.test(model, dataloaders=test_loader)

	
	for batch_idx, input_batch in enumerate(test_loader):
		mix_ri_spec, tgt_ri_spec, doa = input_batch
		est_ri_spec = model.forward(input_batch)

		#breakpoint()
		with torch.no_grad():
			mix_sig = model.format_istft(mix_ri_spec)
			tgt_sig = model.format_istft(tgt_ri_spec)
			est_sig = model.format_istft(est_ri_spec)


			#torch.rad2deg(doa[0,:,1])
			pp_str = f'../signals/{scenario}/{model_train}_model/from_dataset_circular_motion_snr_{snr}_t60_{t60}_src_mic_dist_{src_mic_dist}_noi_mic_dist_{noi_mic_dist}/'    # from_datasetfile_10sec/v2/'
			
			app_str = f'{batch_idx}'
			#app_str = f'sp_{static_prob}_nlp_{non_linear_motion_prob}_snr_{test_snr}_t60_{test_t60}_nb_points_{nb_points}'

			torchaudio.save(f'{pp_str}mix_{app_str}.wav', (mix_sig/torch.max(torch.abs(mix_sig))).cpu(), sample_rate=16000)
			torchaudio.save(f'{pp_str}tgt_{app_str}.wav', (tgt_sig/torch.max(torch.abs(tgt_sig))).cpu(), sample_rate=16000)
			torchaudio.save(f'{pp_str}est_{app_str}.wav', (est_sig/torch.max(torch.abs(est_sig))).cpu(), sample_rate=16000)
			

			mix_metrics = eval_metrics_batch_v1(tgt_sig.cpu().numpy(), mix_sig.cpu().numpy())
			est_metrics = eval_metrics_batch_v1(tgt_sig.cpu().numpy(), est_sig.cpu().numpy())


			#Computing DOA ( F -> T -> F) bcz of vad
			array_setup = model.array_setup
			local_mic_pos = torch.from_numpy(array_setup.mic_pos)
			local_mic_center=torch.from_numpy(np.array([0.0, 0.0, 0.0]))

			mix_utt_doa, mix_utt_sum, mix_f_doa, mix_f_vals, mix_sig_vad, _ = gcc_phat(mix_sig, local_mic_pos = local_mic_pos, local_mic_center=local_mic_center, src_mic_dist=1, weighted=True, is_euclidean_dist=False)
			tgt_utt_doa, tgt_utt_sum, tgt_f_doa, tgt_f_vals, tgt_sig_vad, _ = gcc_phat(tgt_sig, local_mic_pos = local_mic_pos, local_mic_center=local_mic_center, src_mic_dist=1, weighted=True, is_euclidean_dist=False)
			est_utt_doa, est_utt_sum, est_f_doa, est_f_vals, est_sig_vad, _ = gcc_phat(est_sig, local_mic_pos = local_mic_pos, local_mic_center=local_mic_center, src_mic_dist=1, weighted=True, is_euclidean_dist=False)
			

			torch.save( {'mix': (mix_f_doa, mix_f_vals, mix_sig_vad, mix_utt_doa, mix_utt_sum),
						'tgt': (tgt_f_doa, tgt_f_vals, tgt_sig_vad, tgt_utt_doa, tgt_utt_sum),
						'est': (est_f_doa, est_f_vals, est_sig_vad, est_utt_doa, est_utt_sum),
						'lbl_doa': doa,
						'mix_metrics': mix_metrics,
						'est_metrics': est_metrics }, f'{pp_str}doa_{app_str}.pt',
						)

		#break
	