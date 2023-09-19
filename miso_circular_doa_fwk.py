from train import DCCRN_model
from dataset_v2 import MovingSourceDataset, NetworkInput
from array_setup import get_array_set_up_from_config
from masked_gcc_phat import gcc_phat, gcc_phat_v2, compute_vad, block_doa, blk_vad
from metrics import eval_metrics_batch_v1
from callbacks_parallel_doa import DOAcallbacks_parallel

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

class DOA_MISO_fwk_circular(pl.LightningModule):
	
	def __init__(self, array_setup, model_parse_info: list, loss_flags: list):
		super().__init__()
		self.array_setup = array_setup
		
		self.get_model_paths_input(model_parse_info)
		
		self.loss_flags = loss_flags
		

	def get_model_paths_input(self, parse_info: list [list]):
		# assuming [[net_inp, net_out, ref_mic_idx, model_path, loss_wgt_mech],....]
		self.bidirectional = True
	
		self.models = nn.ModuleList()
		for idx, lst_val in enumerate(parse_info):
			for mic_idx, lst in enumerate(lst_val):
				net_inp, net_out, ref_mic_idx, model_path, loss_wgt_mech = lst[0], lst[1], lst[2], lst[3], lst[4]
				
				loss_flag = model_path.split('/')[-4] #4
				if "MISO" not in loss_flag:
					loss_flag = model_path.split('/')[-5]  #5
				
				_model = DCCRN_model.load_from_checkpoint(model_path, bidirectional=self.bidirectional, net_inp=net_inp, net_out=net_out,
							train_dataset=None, val_dataset=None, loss_flag=loss_flag, wgt_mech=loss_wgt_mech)


				self.models.append(_model)

	def _circular_shift_forward(self, input_batch, model, batch_idx):
		mix_ri_spec, tgt_ri_spec, doa = input_batch

		mix_ri_spec_on_cicle =  mix_ri_spec[:,2:,:,:]
		outputs = []
		avg_loss, avg_loss_ri, avg_loss_mag = 0, 0, 0
		for mic_idx in range(1,7):
			rotated_mix_ri_spec_on_circle = torch.roll(mix_ri_spec_on_cicle, shifts=-(mic_idx-1)*2, dims=1)
			#breakpoint()
			rotated_mix_ri_spec = torch.concat((mix_ri_spec[:,:2,:,:], rotated_mix_ri_spec_on_circle), dim=1)

			rotated_input_batch = rotated_mix_ri_spec, tgt_ri_spec[:, 2*mic_idx :2*mic_idx  + 2], doa

			est_ri_spec_dict = model.test_step(rotated_input_batch, batch_idx)

			outputs.append(est_ri_spec_dict["est_ri_spec"])

			#avg_loss
			avg_loss += est_ri_spec_dict["loss"]
			avg_loss_ri += est_ri_spec_dict["loss_ri"]
			avg_loss_mag += est_ri_spec_dict["loss_mag"]

		
		est_ri_spec = torch.concat(outputs, dim=1)

		return est_ri_spec, avg_loss, avg_loss_ri, avg_loss_mag



	def forward(self, input_batch, models, batch_idx):
		mix_ri_spec, tgt_ri_spec, doa = input_batch
		_est_ri_spec = []
		loss_info = []
		num_loss = len(self.loss_flags)
		num_mics = mix_ri_spec.shape[1]/2
		for loss_idx in range(len(self.loss_flags)):
			input_batch_0 = mix_ri_spec, tgt_ri_spec[:,:2], doa
			#breakpoint()
			est_ri_spec_dict = models[num_loss*loss_idx].test_step(input_batch_0, batch_idx)	
			loss_0, loss_ri_0, loss_mag_0 = est_ri_spec_dict["loss"], est_ri_spec_dict["loss_ri"], est_ri_spec_dict["loss_mag"]
		
			est_ri_spec_mic_1_6, avg_loss, avg_loss_ri, avg_loss_mag = self._circular_shift_forward(input_batch, models[num_loss*loss_idx+1], batch_idx)
			
			est_ri_spec = torch.concat((est_ri_spec_dict["est_ri_spec"], est_ri_spec_mic_1_6), dim=1)

			avg_loss += loss_0
			avg_loss_ri += loss_ri_0
			avg_loss_mag += loss_mag_0

			_est_ri_spec.append(est_ri_spec)

			loss_info.append(( avg_loss_ri/num_mics, avg_loss_mag/num_mics)) #avg_loss/num_mics,
			
		est_ri_spec_ = torch.concat(_est_ri_spec)

		return est_ri_spec_, loss_info

	
	def test_step(self, test_batch, batch_idx):
		mix_ri_spec, tgt_ri_spec, doa = test_batch
		est_ri_spec, loss_7mic_info = self.forward(test_batch, self.models, batch_idx)
		return {"est_ri_spec" : est_ri_spec, "loss_info": loss_7mic_info }
	
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

	mic_pairs = [(mic_1, mic_2) for mic_1 in range(0, num_mics) for mic_2 in range(mic_1+1, num_mics)]

	test_dataset = MovingSourceDataset(dataset_file, array_config,# size=5,
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

	app_str = f'{app_str}_circular_miso'
	exp_name = f'Test_{args.exp_name}_{dataset_dtype}_{app_str}' #_loss_functions_comparison_num_mic'
	
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
	
	model = DOA_MISO_fwk_circular(array_config['array_setup'], models_info, loss_list)
	trainer.test(model, dataloaders=test_loader)

if __name__=="__main__":
	#flags
	args = parser.parse_args()
	dataset_dtype = args.dataset_dtype
	dataset_condition = args.dataset_condition
	ref_mic_idx = args.ref_mic_idx
	noise_simulation = args.noise_simulation
	loss_wgt_mech = args.loss_wgt_mech

	print("Testing\n")
	print(f"{torch.cuda.is_available()}, {torch.cuda.device_count()}\n")

	
	loss_list = ["MISO_RI", "MISO_RI_MAG"] 

	models_info = []
	ckpt_dirs = ['/fs/scratch/PAS0774/Shanmukh/ControlledExp/random_seg/Circular_array_MISO/'] 
	for ckpt_dir in ckpt_dirs:
		for _loss_flag in loss_list:
			if '7mic' in ckpt_dir or 'Circular' in ckpt_dir:
				net_inp, net_out = (14,2) 		
				loss_flag_str = f'{_loss_flag}_{loss_wgt_mech}' if ("PD" in _loss_flag) and ("noisy" in dataset_condition) else f'{_loss_flag}'
			models = []
			for ref_mic_idx in [0,1]:
				if "noisy" in dataset_condition:
					ckpt_dir_1 = f'{ckpt_dir}/{loss_flag_str}/{dataset_dtype}/{dataset_condition}/{noise_simulation}/ref_mic_{ref_mic_idx}'
				else:
					ckpt_dir_1 = f'{ckpt_dir}/{loss_flag_str}/{dataset_dtype}/{dataset_condition}/ref_mic_{ref_mic_idx}' 

				if os.path.exists(ckpt_dir_1):
					for _file in os.listdir(ckpt_dir_1):
						if _file.endswith(".ckpt") and _file[:5]=="epoch":
							models.append((net_inp, net_out, ref_mic_idx, os.path.join(ckpt_dir_1, _file), loss_wgt_mech))
							
			models_info.append(models)
	
	#breakpoint()
	test_doa(args, models_info, loss_list)

