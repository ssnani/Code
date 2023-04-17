from train import DCCRN_model
from dataset_v2 import MovingSourceDataset, NetworkInput
from array_setup import get_array_set_up_from_config
from masked_gcc_phat import gcc_phat, gcc_phat_v2, compute_vad, block_doa, blk_vad
from metrics import eval_metrics_batch_v1
from callbacks_parallel import DOAcallbacks_parallel

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

class DOA_MIMO_fwk_num_mics(pl.LightningModule):
	
	def __init__(self, array_setup, model_parse_info: list, loss_flags: list):
		super().__init__()
		self.array_setup = array_setup
		
		self.get_model_paths_input(model_parse_info)
		

	def get_model_paths_input(self, parse_info: list [list]):
		# assuming [[net_inp, net_out, model_path],....]
		self.bidirectional = True
	
		self.models_2mic, self.models_4mic, self.models_8mic = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
		for idx, lst_val in enumerate(parse_info):
			net_inp, net_out, model_path = lst_val[0], lst_val[1], lst_val[2]
			loss_flag = model_path.split('/')[-4]
			if "MIMO" not in loss_flag:
				loss_flag = model_path.split('/')[-5]
			_model = DCCRN_model.load_from_checkpoint(model_path, bidirectional=self.bidirectional, net_inp=net_inp, net_out=net_out,
						   train_dataset=None, val_dataset=None, loss_flag=loss_flag)

			if 4==net_inp:
				self.models_2mic.append(_model)
			elif 8==net_inp:
				self.models_4mic.append(_model)
			elif 16==net_inp:
				self.models_8mic.append(_model)
			else:
				print(f"Incorrect net_inp: {net_inp}")

	def forward(self, input_batch, models, batch_idx):
		mix_ri_spec, tgt_ri_spec, doa = input_batch
		_est_ri_spec = []
		loss_info = []
		for _model in models:
			est_ri_spec_dict = _model.test_step(input_batch, batch_idx)
			_est_ri_spec.append(est_ri_spec_dict['est_ri_spec'])
			loss_ri, loss_mag, loss_ph_diff = est_ri_spec_dict["loss_ri"], est_ri_spec_dict["loss_mag"], est_ri_spec_dict["loss_ph_diff"]
			loss_info.append((loss_ri, loss_mag, loss_ph_diff))
			
		est_ri_spec_ = torch.concat(_est_ri_spec)

		return est_ri_spec_, loss_info

	def forward_v2(self, input_batch, models, batch_idx):
		mix_ri_spec, tgt_ri_spec, doa = input_batch
		_est_ri_spec = []

		for _model in models:
			_model.model.eval()
			with torch.no_grad():
				est_ri_spec = _model.model(mix_ri_spec) #input_batch, batch_idx)
				_est_ri_spec.append(est_ri_spec)
			
		est_ri_spec_ = torch.concat(_est_ri_spec)

		return est_ri_spec_
	
	def test_step(self, test_batch, batch_idx):
		mix_ri_spec, tgt_ri_spec, doa = test_batch

		test_batch_2mic = (mix_ri_spec[:,6:10,:,:], tgt_ri_spec[:,6:10,:,:], doa)
		test_batch_4mic = (mix_ri_spec[:,4:12,:,:], tgt_ri_spec[:,4:12,:,:], doa)

		est_ri_spec_8mic, loss_8mic_info = self.forward(test_batch, self.models_8mic, batch_idx)
		est_ri_spec_4mic, loss_4mic_info = self.forward(test_batch_4mic, self.models_4mic, batch_idx)
		est_ri_spec_2mic, loss_2mic_info = self.forward(test_batch_2mic, self.models_2mic, batch_idx)

		return {"est_8mic_ri_spec" : est_ri_spec_8mic, "est_4mic_ri_spec" : est_ri_spec_4mic, "est_2mic_ri_spec" : est_ri_spec_2mic, "loss_8mic_info": loss_8mic_info, "loss_4mic_info": loss_4mic_info, "loss_2mic_info": loss_2mic_info }
	
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


def test_doa(args, models_info: list, loss_list):

	T = 4
	ref_mic_idx = args.ref_mic_idx
	dataset_file = args.dataset_file
	loss_flag=args.net_type

	# DOA arguments
	doa_tol = args.doa_tol
	doa_euclid_dist = args.doa_euclid_dist
	wgt_mech = args.doa_wgt_mech

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

	test_dataset = MovingSourceDataset(dataset_file, array_config, #size=10,
									transforms=[ NetworkInput(320, 160, ref_mic_idx)],
									T60=T60, SNR=SNR, dataset_dtype=dataset_dtype, dataset_condition=dataset_condition,
									noise_simulation=noise_simulation, diffuse_files_path=diffuse_files_path) #
	test_loader = DataLoader(test_dataset, batch_size = args.batch_size,
							 num_workers=args.num_workers, pin_memory=True, drop_last=True)  

	## exp path director

	
	#ckpt_dir = f'{args.ckpt_dir}/{dataset_dtype}/{dataset_condition}/ref_mic_{ref_mic_idx}'
	

	if args.dataset_condition =="reverb":
		app_str = f't60_{T60}_comparison'
		#ckpt_dir_0 = f'{args.ckpt_dir}/{model_0_loss_flag}/{dataset_dtype}/{dataset_condition}/ref_mic_{ref_mic_idx}'
		#ckpt_dir_1 = f'{args.ckpt_dir}/{model_1_loss_flag}/{dataset_dtype}/{dataset_condition}/ref_mic_{ref_mic_idx}' #f'{args.ckpt_dir}/{dataset_condition}/ref_mic_{ref_mic_idx}'   #{noise_simulation}/
		ckpt_dir = f'{args.ckpt_dir}'
	elif args.dataset_condition =="noisy":
		app_str = f'snr_{SNR}dB'
		ckpt_dir = f'{args.ckpt_dir}'
	elif args.dataset_condition =="noisy_reverb":
		app_str = f't60_{T60}_snr_{SNR}dB'
		ckpt_dir = f'{args.ckpt_dir}' # /{dataset_dtype}/{dataset_condition}/ref_mic_{ref_mic_idx}' #{noise_simulation}/
	else:
		app_str = ''

	exp_name = f'Test_{args.exp_name}_{dataset_dtype}_{app_str}_loss_functions_comparison_num_mic'
	
	tb_logger = pl_loggers.TensorBoardLogger(save_dir=ckpt_dir, version=exp_name)
	precision = 32
	trainer = pl.Trainer(accelerator='gpu', precision=precision, devices=args.num_gpu_per_node, num_nodes=args.num_nodes,
						callbacks=[ DOAcallbacks_parallel(array_config=array_config, dataset_condition = dataset_condition, noise_simulation = noise_simulation, 
			       						doa_tol=doa_tol, doa_euclid_dist=doa_euclid_dist, mic_pairs=mic_pairs, wgt_mech=wgt_mech, loss_flags=loss_list, log_str = app_str, dbg_doa_log=False)], #Losscallbacks(),
						logger=tb_logger
						)
	bidirectional = args.bidirectional
	
	
	msg = f"Test Config: bidirectional: {bidirectional}, T: {T}, batch_size: {args.batch_size}, precision: {precision}, \n \
		ckpt_dir: {args.ckpt_dir}, exp_name: {exp_name}, \n \
		net_inp: {net_inp}, net_out: {net_out}, \n \
		models: {models_info} ref_mic_idx : {ref_mic_idx}, \n \
		dataset_file: {dataset_file}, t60: {T60}, snr: {SNR}, dataset_dtype: {dataset_dtype}, dataset_condition: {dataset_condition}, \n\
		doa_tol: {doa_tol}, doa_euclid_dist: {doa_euclid_dist}, doa_wgt_mech: {wgt_mech} \n"

	trainer.logger.experiment.add_text("Exp details", msg)

	print(msg)
	
	model = DOA_MIMO_fwk_num_mics(array_config['array_setup'], models_info, loss_list)
	trainer.test(model, dataloaders=test_loader)

if __name__=="__main__":
	#flags
	args = parser.parse_args()
	dataset_dtype = args.dataset_dtype
	dataset_condition = args.dataset_condition
	ref_mic_idx = args.ref_mic_idx
	noise_simulation = args.noise_simulation

	print("Testing\n")
	print(f"{torch.cuda.is_available()}, {torch.cuda.device_count()}\n")

	
	loss_list = ["MIMO_RI", "MIMO_RI_MAG", "MIMO_RI_MAG_PD", "MIMO_RI_PD", "MIMO_RI_PD_REF"]
	#loss_list = ["MIMO_RI_MAG_PD", "MIMO_RI"]
	#if '4mic' in args.ckpt_dir:
	#	loss_list.append("MIMO_RI_PD_REF")
	models_info = []
	ckpt_dirs = ['/fs/scratch/PAS0774/Shanmukh/ControlledExp/random_seg/Linear_array_8cm_dp_rir_t60_0', '/fs/scratch/PAS0774/Shanmukh/ControlledExp/random_seg/Linear_array_8cm_4mic_dp_rir_t60_0', '/fs/scratch/PAS0774/Shanmukh/ControlledExp/random_seg/Linear_array_8cm_8mic_dp_rir_t60_0']
	for ckpt_dir in ckpt_dirs:
		if '8mic' in ckpt_dir:
			net_inp, net_out = (16,16) 
		elif '4mic' in ckpt_dir:
			net_inp, net_out = (8,8) 
		else:
			net_inp, net_out = (4,4)

		for _loss_flag in loss_list:
			ckpt_dir_1 = f'{ckpt_dir}/{_loss_flag}/{dataset_dtype}/{dataset_condition}/ref_mic_{ref_mic_idx}' #{noise_simulation}/
			if os.path.exists(ckpt_dir_1):
				for _file in os.listdir(ckpt_dir_1):
					if _file.endswith(".ckpt") and _file[:5]=="epoch":
						models_info.append((net_inp,net_out , os.path.join(ckpt_dir_1, _file)))

	#breakpoint()
	test_doa(args, models_info, loss_list)

