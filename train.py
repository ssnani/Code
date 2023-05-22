import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from pytorch_lightning.profiler import SimpleProfiler
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, ModelSummary, LearningRateMonitor

import os
import sys 
import random
from dataset_v2 import MovingSourceDataset, NetworkInput
from array_setup import get_array_set_up_from_config
from networks import Net, MIMO_Net
from loss_criterion import LossFunction, MIMO_LossFunction
from arg_parser import parser
from debug import dbg_print
from callbacks import Losscallbacks, DOAcallbacks, GradNormCallback

class DCCRN_model(pl.LightningModule):
	def __init__(self, bidirectional: bool, net_inp: int, net_out: int, train_dataset: Dataset, val_dataset: Dataset, batch_size=32, num_workers=4, loss_flag=None, wgt_mech=None):
		super().__init__()
		pl.seed_everything(77)

		self.model = MIMO_Net(bidirectional, net_inp, net_out) #MIMO_Net(bidirectional) if "MIMO" in loss_flag else Net(bidirectional) # in 
		
		self.loss = MIMO_LossFunction(loss_flag, wgt_mech, net_out) if "MIMO" in loss_flag else LossFunction(loss_flag)

		self.batch_size = batch_size
		self.num_workers = num_workers
		self.train_dataset = train_dataset
		self.val_dataset = val_dataset

		self.loss_flag = loss_flag
		self.wgt_mech = wgt_mech


	def train_dataloader(self):
		return DataLoader(self.train_dataset, batch_size = self.batch_size, 
							 num_workers=self.num_workers, pin_memory=True, drop_last=True)

	def val_dataloader(self):		
		return DataLoader(self.val_dataset, batch_size = self.batch_size, 
							 num_workers=self.num_workers, pin_memory=True, drop_last=True)

	def configure_optimizers(self):
		_optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, amsgrad=True)
		_lr_scheduler = torch.optim.lr_scheduler.StepLR(_optimizer, step_size=2, gamma=0.98)
		#_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(_optimizer, factor=0.5, patience=5)
		return {"optimizer": _optimizer, "lr_scheduler": _lr_scheduler, "monitor": 'val_loss'}

	def forward(self, input_batch):
		mix_ri_spec, tgt_ri_spec, doa = input_batch
		est_ri_spec = self.model(mix_ri_spec)
		return est_ri_spec, tgt_ri_spec, mix_ri_spec

	def training_step(self, train_batch, batch_idx):

		est_ri_spec, tgt_ri_spec, mix_ri_spec = self.forward(train_batch)
		
		if "MIMO" in self.loss_flag:
			loss, loss_ri, loss_mag, loss_ph_diff, loss_mag_diff = self.loss(est_ri_spec, tgt_ri_spec, mix_ri_spec)
		else:
			loss, loss_ri, loss_mag, loss_ph = self.loss(est_ri_spec, tgt_ri_spec)

		self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True )
		self.log('loss_ri', loss_ri, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True )
		self.log('loss_mag', loss_mag, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True )	
		if "MIMO" in self.loss_flag:
			self.log('loss_ph_diff', loss_ph_diff, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True )
			self.log('loss_mag_diff', loss_mag_diff, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True )
			return {"loss" : loss , "loss_ri": loss_ri, "loss_mag": loss_mag, "loss_ph_diff": loss_ph_diff, 'loss_mag_diff': loss_mag_diff, "est_ri_spec" : est_ri_spec }
		else:
			self.log('loss_ph', loss_ph, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True )
			return {"loss" : loss , "loss_ri": loss_ri, "loss_mag": loss_mag, 'loss_ph': loss_ph, "est_ri_spec" : est_ri_spec } 


	def validation_step(self, val_batch, batch_idx):
		est_ri_spec, tgt_ri_spec, mix_ri_spec = self.forward(val_batch)
		
		if "MIMO" in self.loss_flag:
			loss, loss_ri, loss_mag, loss_ph_diff, loss_mag_diff = self.loss(est_ri_spec, tgt_ri_spec, mix_ri_spec)
		else:
			loss, loss_ri, loss_mag, loss_ph  = self.loss(est_ri_spec, tgt_ri_spec) 

		self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True )
		self.log('val_loss_ri', loss_ri, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True )
		self.log('val_loss_mag', loss_mag, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True )
		if "MIMO" in self.loss_flag:
			self.log('val_loss_ph_diff', loss_ph_diff, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True )
			self.log('val_loss_mag_diff', loss_mag_diff, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True )
			return {"loss" : loss , "loss_ri": loss_ri, "loss_mag": loss_mag, "loss_ph_diff": loss_ph_diff, 'loss_mag_diff': loss_mag_diff, "est_ri_spec" : est_ri_spec }
		else:
			self.log('val_loss_ph', loss_ph, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True )
			return {"loss" : loss , "loss_ri": loss_ri, "loss_mag": loss_mag, 'loss_ph': loss_ph, "est_ri_spec" : est_ri_spec } 


	def test_step(self, test_batch, batch_idx):
		est_ri_spec, tgt_ri_spec, mix_ri_spec = self.forward(test_batch)

		if "MIMO" in self.loss_flag:
			loss, loss_ri, loss_mag, loss_ph_diff, loss_mag_diff = self.loss(est_ri_spec, tgt_ri_spec, mix_ri_spec)
		else:
			loss, loss_ri, loss_mag, loss_ph  = self.loss(est_ri_spec, tgt_ri_spec) 

		self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True )
		self.log('test_loss_ri', loss_ri, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True )
		self.log('test_loss_mag', loss_mag, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True )
		if "MIMO" in self.loss_flag:
			self.log('test_loss_ph_diff', loss_ph_diff, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True )
			self.log('test_loss_mag_diff', loss_mag_diff, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True )
			return {"loss" : loss , "loss_ri": loss_ri, "loss_mag": loss_mag, "loss_ph_diff": loss_ph_diff, 'loss_mag_diff': loss_mag_diff, "est_ri_spec" : est_ri_spec }

		else:
			self.log('test_loss_ph', loss_ph, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True )
			return {"loss" : loss , "loss_ri": loss_ri, "loss_mag": loss_mag, 'loss_ph': loss_ph, "est_ri_spec" : est_ri_spec }


	def validation_epoch_end(self, validation_step_outputs):
		tensorboard = self.logger.experiment
		if (self.current_epoch %10==0):
			for batch_idx, batch_output in enumerate(validation_step_outputs):
				#if ((batch_idx+self.current_epoch)%4==0):
				idx = random.randint(0, self.batch_size-1)
				est_ri_spec = batch_output['est_ri_spec'][idx]
				est_ri_spec = est_ri_spec.to(torch.float32)
				(n_ch, n_frms, n_freq) = est_ri_spec.shape #n_ch (ri*num_ch)
				if n_ch>2:
					est_ri_spec = torch.reshape(est_ri_spec,(n_ch//2, 2, n_frms, n_freq))
					_est_ri_spec = torch.permute(est_ri_spec,[0,3,2,1])  #(2,T,F) -> (F,T,2)
				else:
					_est_ri_spec = torch.permute(est_ri_spec,[2,1,0])  #(2,T,F) -> (F,T,2)
				est_sig = torch.istft(_est_ri_spec, 320,160,320,torch.hamming_window(320).type_as(_est_ri_spec))
				
				if len(est_sig.shape)> 1:
					n_sig, _ = est_sig.shape
				else:
					n_sig=1

				if n_sig>1:
					#rank_zero_only
					for i in range(n_sig):
						tensorboard.add_audio(f'est_{self.current_epoch}_{batch_idx}_{i}', est_sig[[i],:]/torch.max(torch.abs(est_sig[[i],:])), sample_rate=16000)
					#tensorboard.add_audio(f'est_{self.current_epoch}_{batch_idx}_1', est_sig[[1],:]/torch.max(torch.abs(est_sig[[1],:])), sample_rate=16000)
				else:
					tensorboard.add_audio(f'est_{self.current_epoch}_{batch_idx}', est_sig/torch.max(torch.abs(est_sig)), sample_rate=16000)

def main(args):
	dbg_print(f"{torch.cuda.is_available()}, {torch.cuda.device_count()}\n")
	T = 4
	#array jobs code changes
	#reading from file for array jobs
	if args.array_job:

		T60 = None
		SNR = None
		with open(args.input_train_filename, 'r') as f:
			lines = [line for line in f.readlines()]

		array_config = {}
		#key value 
		for line in lines:
			lst = line.strip().split()
			if lst[0]=="dataset_dtype":
				dataset_dtype = lst[1]
			elif lst[0]=="dataset_condition":
				dataset_condition = lst[1]
			elif lst[0]=="ref_mic_idx":
				ref_mic_idx = int(lst[1])
			elif lst[0]=="dataset_file":
				dataset_file = lst[1]
			elif lst[0]=="val_dataset_file":
				val_dataset_file = lst[1]
			elif lst[0]=="array_type":
				array_config['array_type'] = lst[1]
			elif lst[0]=="num_mics":
				array_config['num_mics'] = int(lst[1])
			elif lst[0]=="intermic_dist":
				array_config['intermic_dist'] = float(lst[1])
			elif lst[0]=="room_size":
				array_config['room_size'] = [lst[1], lst[2], lst[3]]
			elif lst[0]=="loss":
				loss_flag = lst[1]
			else:
				continue
	
	else:
		ref_mic_idx = args.ref_mic_idx
		T60 = args.T60 
		SNR = args.SNR

		dataset_dtype = args.dataset_dtype
		dataset_condition = args.dataset_condition
		#Loading datasets
		#scenario = args.scenario
		
		dataset_file = args.dataset_file #f'../dataset_file_circular_{scenario}_snr_{snr}_t60_{t60}.txt'
		val_dataset_file = args.val_dataset_file #f'../dataset_file_circular_{scenario}_snr_{snr}_t60_{t60}.txt'

	noise_simulation = args.noise_simulation
	diffuse_files_path = args.diffuse_files_path
	loss_wgt_mech = args.loss_wgt_mech
	#loss_flag
	net_inp = array_config["num_mics"]*2

	if "SISO" in loss_flag:
		net_inp, net_out = 2, 2
	elif "MISO" in loss_flag:
		net_inp, net_out = net_inp, 2
	elif "MIMO" in loss_flag:
		net_inp, net_out = net_inp, net_inp

	
	array_config['array_setup'] = get_array_set_up_from_config(array_config['array_type'], array_config['num_mics'], array_config['intermic_dist'])
	train_dataset = MovingSourceDataset(dataset_file, array_config, transforms=[ NetworkInput(320, 160, ref_mic_idx, array_config['array_type'])], 
										T60=T60, SNR=SNR, dataset_dtype=dataset_dtype, dataset_condition=dataset_condition, train_flag=args.train,
										noise_simulation=noise_simulation, diffuse_files_path=diffuse_files_path) #, size=20)

	dev_dataset = MovingSourceDataset(val_dataset_file, array_config, transforms=[ NetworkInput(320, 160, ref_mic_idx, array_config['array_type'])],
									T60=T60, SNR=SNR, dataset_dtype=dataset_dtype, dataset_condition=dataset_condition, train_flag=args.train,
									noise_simulation=noise_simulation, diffuse_files_path=diffuse_files_path) #, size=20


	# model
	bidirectional = args.bidirectional
	model = DCCRN_model(bidirectional, net_inp, net_out, train_dataset, dev_dataset, args.batch_size, args.num_workers, loss_flag, loss_wgt_mech)


	## exp path directories
	if dataset_condition=="reverb":
		ckpt_dir = f'{args.ckpt_dir}/{loss_flag}/{dataset_dtype}/{dataset_condition}/ref_mic_{ref_mic_idx}'
	else:
		loss_flag_str = f'{loss_flag}_{loss_wgt_mech}' if "PD" in loss_flag else f'{loss_flag}'
		ckpt_dir = f'{args.ckpt_dir}/{loss_flag_str}/{dataset_dtype}/{dataset_condition}/{noise_simulation}/ref_mic_{ref_mic_idx}'
	exp_name = f'{args.exp_name}' #t60_{T60}_snr_{SNR}dB

	msg_pre_trained = None
	if (not os.path.exists(os.path.join(ckpt_dir,args.resume_model))) and len(args.pre_trained_ckpt_path)>0:
		msg_pre_trained = f"Loading ONLY model parameters from {args.pre_trained_ckpt_path}"
		print(msg_pre_trained)
		ckpt_point = torch.load(args.pre_trained_ckpt_path)
		model.load_state_dict(ckpt_point['state_dict'],strict=False) 

	tb_logger = pl_loggers.TensorBoardLogger(save_dir=ckpt_dir, version=exp_name)
	checkpoint_callback = ModelCheckpoint(dirpath=ckpt_dir, save_last = True, save_top_k=1, monitor='val_loss')

	#pesq_checkpoint_callback = ModelCheckpoint(dirpath=ckpt_dir, filename='{epoch}-{VAL_PESQ_NB:.2f}--{val_loss:.2f}-{VAL_STOI:.2f}', save_last = True, save_top_k=1, monitor='PESQ_NB')
	#pesq_checkpoint_callback.CHECKPOINT_NAME_LAST = "pesq-nb-last"
	#stoi_checkpoint_callback = ModelCheckpoint(dirpath=ckpt_dir, filename='{epoch}-{VAL_STOI:.2f}-{val_loss:.2f}-{VAL_PESQ_NB:.2f}', save_last = True, save_top_k=1, monitor='STOI')
	#stoi_checkpoint_callback.CHECKPOINT_NAME_LAST = "stoi-last"

	model_summary = ModelSummary(max_depth=1)
	early_stopping = EarlyStopping('val_loss', patience=10)
	lr_monitor = LearningRateMonitor(logging_interval='step')

	# training
	precision=16*2
	trainer = pl.Trainer(accelerator='gpu', devices=args.num_gpu_per_node, num_nodes=args.num_nodes, precision=precision,
					max_epochs = args.max_n_epochs,
					callbacks=[checkpoint_callback, model_summary, lr_monitor],# pesq_checkpoint_callback, stoi_checkpoint_callback, DOAcallbacks()], early_stopping, GradNormCallback()
					logger=tb_logger,
					strategy="ddp_find_unused_parameters_false",
					check_val_every_n_epoch=1,
					log_every_n_steps = 1,
					num_sanity_val_steps=-1,
					profiler="simple",
					fast_dev_run=args.fast_dev_run,
					auto_scale_batch_size=False,
					detect_anomaly=True,
					#gradient_clip_val=5
					)
	
	#trainer.tune(model)
	#print(f'Max batch size fit on memory: {model.batch_size}\n')
				
	msg = f"Train Config: bidirectional: {bidirectional}, net_inp: {net_inp}, net_out: {net_out}, T: {T} , loss_flag: {loss_flag}, loss_wgt_mech: {loss_wgt_mech}, precision: {precision}, \n \
		array_type: {array_config['array_type']}, num_mics: {array_config['num_mics']}, intermic_dist: {array_config['intermic_dist']}, room_size: {array_config['room_size']} \n, \
		dataset_file: {dataset_file}, t60: {T60}, snr: {SNR}, dataset_dtype: {dataset_dtype}, dataset_condition: {dataset_condition}, \n \
		ref_mic_idx: {ref_mic_idx}, batch_size: {args.batch_size}, ckpt_dir: {ckpt_dir}, exp_name: {exp_name} \n"

	trainer.logger.experiment.add_text("Exp details", msg)

	print(msg)
	if os.path.exists(os.path.join(ckpt_dir,args.resume_model)):
		trainer.fit(model, ckpt_path=os.path.join(ckpt_dir,args.resume_model)) #train_loader, val_loader,
	else:
		trainer.fit(model)#, train_loader, val_loader)

def test(args):

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

	
	#if "MIMO_RI_PD_REF" == loss_flag:
	#	mic_1 = 0
	#	mic_pairs = [(mic_1, mic_2) for mic_2 in range(mic_1+1, num_mics)]
	#else:	
	mic_pairs = [(mic_1, mic_2) for mic_1 in range(0, num_mics) for mic_2 in range(mic_1+1, num_mics)]

	test_dataset = MovingSourceDataset(dataset_file, array_config, #size=1,
									transforms=[ NetworkInput(320, 160, ref_mic_idx)],
									T60=T60, SNR=SNR, dataset_dtype=dataset_dtype, dataset_condition=dataset_condition,
									noise_simulation=noise_simulation, diffuse_files_path=diffuse_files_path) #
	test_loader = DataLoader(test_dataset, batch_size = args.batch_size,
							 num_workers=args.num_workers, pin_memory=True, drop_last=True)  

	## exp path directories

	#ckpt_dir = f'{args.ckpt_dir}/{dataset_dtype}/{dataset_condition}/ref_mic_{ref_mic_idx}'
	

	if args.dataset_condition =="reverb":
		app_str = f't60_{T60}'
		ckpt_dir = f'{args.ckpt_dir}/{loss_flag}/{dataset_dtype}/{dataset_condition}/ref_mic_{ref_mic_idx}'   #{noise_simulation}/
	elif args.dataset_condition =="noisy":
		app_str = f'snr_{SNR}dB'
		ckpt_dir = f'{args.ckpt_dir}/{loss_flag}/{dataset_dtype}/{dataset_condition}/{noise_simulation}/ref_mic_{ref_mic_idx}'  #{noise_simulation}
	elif args.dataset_condition =="noisy_reverb":
		app_str = f't60_{T60}_snr_{SNR}dB'
		ckpt_dir = f'{args.ckpt_dir}/{loss_flag}/{dataset_dtype}/{dataset_condition}/{noise_simulation}/ref_mic_{ref_mic_idx}'  #{noise_simulation}
	else:
		app_str = ''

	exp_name = f'Test_{args.exp_name}_{dataset_dtype}_{app_str}'
	
	tb_logger = pl_loggers.TensorBoardLogger(save_dir=ckpt_dir, version=exp_name)
	precision = 32
	trainer = pl.Trainer(accelerator='gpu', precision=precision, devices=args.num_gpu_per_node, num_nodes=args.num_nodes,
						callbacks=[ DOAcallbacks(array_config=array_config, doa_tol=doa_tol, doa_euclid_dist=doa_euclid_dist, mic_pairs=mic_pairs, wgt_mech=wgt_mech)], #Losscallbacks(),
						logger=tb_logger
						)
	bidirectional = args.bidirectional

	#getting model
	for _file in os.listdir(ckpt_dir):
		if _file.endswith(".ckpt") and _file[:5]=="epoch":
			model_path = _file

	
	msg = f"Test Config: bidirectional: {bidirectional}, T: {T}, batch_size: {args.batch_size}, precision: {precision}, \n \
		ckpt_dir: {ckpt_dir}, exp_name: {exp_name}, \n \
		net_inp: {net_inp}, net_out: {net_out}, \n \
		model: {model_path}, ref_mic_idx : {ref_mic_idx}, \n \
		dataset_file: {dataset_file}, t60: {T60}, snr: {SNR}, dataset_dtype: {dataset_dtype}, dataset_condition: {dataset_condition}, \n\
		doa_tol: {doa_tol}, doa_euclid_dist: {doa_euclid_dist}, doa_wgt_mech: {wgt_mech} \n"

	trainer.logger.experiment.add_text("Exp details", msg)

	print(msg)

	if os.path.exists(os.path.join(ckpt_dir, model_path)):  #args.
		model = DCCRN_model.load_from_checkpoint(os.path.join(ckpt_dir, model_path), bidirectional=bidirectional, #args.
					   							net_inp=net_inp, net_out=net_out,
		        								train_dataset=None, val_dataset=None, loss_flag=loss_flag)

		trainer.test(model, dataloaders=test_loader)
	else:
		print(f"Model path not found in {ckpt_dir}")

if __name__=="__main__":
	#flags
	#torch.autograd.set_detect_anomaly(True)
	args = parser.parse_args()
	if args.train:
		print(f"{torch.cuda.is_available()}, {torch.cuda.device_count()}\n")
		main(args)
	else:
		print("Testing\n")
		print(f"{torch.cuda.is_available()}, {torch.cuda.device_count()}\n")
		test(args)
