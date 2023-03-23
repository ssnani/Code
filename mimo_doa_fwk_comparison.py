from train import DCCRN_model
from dataset_v2 import MovingSourceDataset, NetworkInput
from array_setup import get_array_set_up_from_config
from masked_gcc_phat import gcc_phat, gcc_phat_v2, compute_vad, block_doa, blk_vad
from metrics import eval_metrics_batch_v1
from callbacks import DOAcallbacks

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
import os
from arg_parser import parser

class DOA_MIMO_fwk(pl.LightningModule):
	
	def __init__(self, array_setup, net_inp: int, net_out: int, model_paths: list):
		super().__init__()
		self.array_setup = array_setup
		self.bidirectional = True
	
		self.model_paths = model_paths 
		self.models = [ DCCRN_model.load_from_checkpoint(model_path, bidirectional=self.bidirectional,  net_inp=net_inp, net_out=net_out,
						   train_dataset=None, val_dataset=None, loss_flag="MIMO") 
		              for model_path in self.model_paths]

	
	def forward(self, input_batch):
		mix_ri_spec, tgt_ri_spec, doa = input_batch
		_est_ri_spec = []
		for _model in self.models:
			_model.cuda()
			est_ri_spec, _ = _model(input_batch)
			_est_ri_spec.append(est_ri_spec)
		est_ri_spec = torch.concat(_est_ri_spec)

		return est_ri_spec

	def test_step(self, test_batch, batch_idx):
		est_ri_spec = self.forward(test_batch)
		return {"est_ri_spec" : est_ri_spec }
	
	def training_step(self, test_batch, batch_idx):
		pass

	def validation_step(self, test_batch, batch_idx):
		pass

	def configure_optimizers(self):
		pass

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


def test_doa(args, loss_flags: list):

	T = 4
	ref_mic_idx = args.ref_mic_idx
	dataset_file = args.dataset_file
	loss_flag=args.net_type

	# DOA arguments
	doa_tol = args.doa_tol
	doa_euclid_dist = args.doa_euclid_dist

	if 0:
		T60 = args.T60 
		SNR = args.SNR
		
	else:
		#reading from file for array jobs
		with open(args.input_test_filename, 'r') as f:
			lines = [line for line in f.readlines()]

		T60 = None
		SNR = None
		array_config = {}
		#key value 
		for line in lines:
			lst = line.strip().split()
			if lst[0]=="T60":
				T60 = float(lst[1])
			elif lst[0]=="SNR":
				SNR = float(lst[1])
			elif lst[0]=="array_type":
				array_config['array_type'] = lst[1]
			elif lst[0]=="num_mics":
				array_config['num_mics'] = int(lst[1])
			elif lst[0]=="intermic_dist":
				array_config['intermic_dist'] = float(lst[1])
			elif lst[0]=="room_size":
				array_config['room_size'] = [lst[1], lst[2], lst[3]]
			elif lst[0]=="real_rirs":
				array_config["real_rirs"] = True
			elif lst[0]=="dist":
				array_config["dist"] = int(lst[1])
			else:
				continue
	
	dataset_condition = args.dataset_condition
	dataset_dtype = args.dataset_dtype
	noise_simulation = args.noise_simulation
	diffuse_files_path = args.diffuse_files_path
	array_config['array_setup'] = get_array_set_up_from_config(array_config['array_type'], array_config['num_mics'], array_config['intermic_dist'])
	
	net_inp = array_config["num_mics"]*2

	net_inp, net_out = net_inp, net_inp

	num_mics = array_config["num_mics"]

	"""
	if "MIMO_RI_PD_REF" == loss_flag:
		mic_1 = 0
		mic_pairs = [(mic_1, mic_2) for mic_2 in range(mic_1+1, num_mics)]
	else:	
	"""
	mic_pairs = [(mic_1, mic_2) for mic_1 in range(0, num_mics) for mic_2 in range(mic_1+1, num_mics)]

	test_dataset = MovingSourceDataset(dataset_file, array_config, size=5,
									transforms=[ NetworkInput(320, 160, ref_mic_idx)],
									T60=T60, SNR=SNR, dataset_dtype=dataset_dtype, dataset_condition=dataset_condition,
									noise_simulation=noise_simulation, diffuse_files_path=diffuse_files_path) #
	test_loader = DataLoader(test_dataset, batch_size = args.batch_size,
							 num_workers=args.num_workers, pin_memory=True, drop_last=True)  

	## exp path directories
	model_paths = []

	for _loss_flag in loss_flags:
		ckpt_dir_1 = f'{args.ckpt_dir}/{_loss_flag}/{dataset_dtype}/{dataset_condition}/ref_mic_{ref_mic_idx}'
		for _file in os.listdir(ckpt_dir_1):
			if _file.endswith(".ckpt") and _file[:5]=="epoch":
				model_paths.append(os.path.join(ckpt_dir_1, _file))


	#ckpt_dir = f'{args.ckpt_dir}/{dataset_dtype}/{dataset_condition}/ref_mic_{ref_mic_idx}'
	

	if args.dataset_condition =="reverb":
		app_str = f't60_{T60}_comparison'
		ckpt_dir_0 = f'{args.ckpt_dir}/{model_0_loss_flag}/{dataset_dtype}/{dataset_condition}/ref_mic_{ref_mic_idx}'
		ckpt_dir_1 = f'{args.ckpt_dir}/{model_1_loss_flag}/{dataset_dtype}/{dataset_condition}/ref_mic_{ref_mic_idx}' #f'{args.ckpt_dir}/{dataset_condition}/ref_mic_{ref_mic_idx}'   #{noise_simulation}/
	elif args.dataset_condition =="noisy":
		app_str = f'snr_{SNR}dB'
	elif args.dataset_condition =="noisy_reverb":
		app_str = f't60_{T60}_snr_{SNR}dB'
		ckpt_dir = f'{args.ckpt_dir}' # /{dataset_dtype}/{dataset_condition}/ref_mic_{ref_mic_idx}' #{noise_simulation}/
	else:
		app_str = ''

	exp_name = f'Test_{args.exp_name}_{dataset_dtype}_{app_str}_loss_functions_comparison'
	
	tb_logger = pl_loggers.TensorBoardLogger(save_dir=ckpt_dir, version=exp_name)
	precision = 32
	trainer = pl.Trainer(accelerator='gpu', precision=precision, devices=args.num_gpu_per_node, num_nodes=args.num_nodes,
						callbacks=[ DOAcallbacks(array_config=array_config, doa_tol=doa_tol, doa_euclid_dist=doa_euclid_dist, mic_pairs=mic_pairs)], #Losscallbacks(),
						logger=tb_logger
						)
	bidirectional = args.bidirectional
	
	
	msg = f"Test Config: bidirectional: {bidirectional}, T: {T}, batch_size: {args.batch_size}, precision: {precision}, \n \
		ckpt_dir: {args.ckpt_dir}, exp_name: {exp_name}, \n \
		net_inp: {net_inp}, net_out: {net_out}, \n \
		models: {model_paths} ref_mic_idx : {ref_mic_idx}, \n \
		dataset_file: {dataset_file}, t60: {T60}, snr: {SNR}, dataset_dtype: {dataset_dtype}, dataset_condition: {dataset_condition}, \n\
		doa_tol: {doa_tol}, doa_euclid_dist: {doa_euclid_dist} \n"

	trainer.logger.experiment.add_text("Exp details", msg)

	print(msg)
	
	model = DOA_MIMO_fwk(array_config['array_setup'], net_inp, net_out, model_paths)
	trainer.test(model, dataloaders=test_loader)

if __name__=="__main__":
	#flags
	args = parser.parse_args()

	print("Testing\n")
	print(f"{torch.cuda.is_available()}, {torch.cuda.device_count()}\n")
	loss_list = ["MIMO_RI", "MIMO_RI_MAG", "MIMO_RI_MAG_PD", "MIMO_RI_PD", "MIMO_RI_PD_REF"]
	test_doa(args, loss_list)

