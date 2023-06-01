
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

from arg_parser import parser
from ZQ_dataset import ZQ_NetworkInput
from dataset_v2 import MovingSourceDataset
from array_setup import get_array_set_up_from_config
from loss_criterion import MSELoss
from callbacks import DOAcallbacks

#from metrics import gettdsignal_mag_spec_mask, _my_istft
#from debug_flags import _control_flags
#from config import Config

class ZQRNN(pl.LightningModule):
	def __init__(self, bidirectional, train_dataset: Dataset, val_dataset: Dataset, batch_size=32, num_workers=4, loss_flag:str = "PSM_MSE"):
		super().__init__()
		self.inp_size = 161
		self.out_size = 161
		self.num_hid_lyrs = 2
		self.hid_size = 600
		self.bidirectional = bidirectional
		self.dropout = 0.1
		self.rnn = nn.LSTM( self.inp_size, self.hid_size, self.num_hid_lyrs, 
							batch_first=True, dropout=self.dropout,
							bidirectional=self.bidirectional
							) 
							
		self.rnn_out_size = self.hid_size*2 if self.bidirectional else self.hid_size
		self.mask_lin_lyr = nn.Linear(self.rnn_out_size, self.out_size)
		self.loss = MSELoss()

		self.batch_size = batch_size
		self.num_workers = num_workers
		self.train_dataset = train_dataset
		self.val_dataset = val_dataset
		

	def train_dataloader(self):
		return DataLoader(self.train_dataset, batch_size = self.batch_size, 
							 num_workers=self.num_workers, pin_memory=True, drop_last=True)

	def val_dataloader(self):		
		return DataLoader(self.val_dataset, batch_size = self.batch_size, 
							 num_workers=self.num_workers, pin_memory=True, drop_last=True)

	def configure_optimizers(self):
		_optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
		_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(_optimizer, mode='min',factor=0.5, patience=3)
		return {"optimizer": _optimizer, "lr_scheduler": _lr_scheduler, "monitor": 'val_loss'}
	
	def _init_hidden_states(self, batch_size):
		D = 2 if self.bidirectional else 1
		return torch.zeros(self.num_hid_lyrs*D, batch_size, self.hid_size, device=self.device)
	
	def forward(self, log_ps, seq_len):
		batch_size, frms, freq = log_ps.shape
		
		h_0, c_0 = self._init_hidden_states(batch_size), self._init_hidden_states(batch_size)
		#breakpoint()
		_packed_log_ps = torch.nn.utils.rnn.pack_padded_sequence(log_ps, seq_len.cpu(), batch_first=True, enforce_sorted=False)
		_packed_hid_t, _ = self.rnn(_packed_log_ps, (h_0, c_0)) 
		hid_t, _seq_len = nn.utils.rnn.pad_packed_sequence(_packed_hid_t, batch_first=True)

		est_mask = torch.sigmoid(self.mask_lin_lyr(hid_t))

		return est_mask

	def causal_forward(self, log_ps, seq_len):
		batch_size, _, _ = log_ps.squeeze().shape
		h_0, c_0 = self._init_hidden_states(batch_size), self._init_hidden_states(batch_size)

		_packed_log_ps = torch.nn.utils.rnn.pack_padded_sequence(log_ps, seq_len.cpu(), batch_first=True)
		_packed_hid_t, _ = self.rnn(_packed_log_ps, (h_0, c_0)) 
		hid_t, _seq_len = nn.utils.rnn.pad_packed_sequence(_packed_hid_t, batch_first=True)

		## Incorporate Zeros for backward lstm
		#breakpoint()
		out_dim_sh = hid_t.shape
		hid_t[:,:,self.hid_size:self.rnn_out_size] = torch.zeros(out_dim_sh[0],out_dim_sh[1], self.hid_size)
		est_mask = torch.sigmoid(self.mask_lin_lyr(hid_t))

		return est_mask

	def training_step(self, train_batch, batch_idx):
		log_ps, tgt_mask, seq_len, _, _ = train_batch

		batch_size, num_mics, frms, freq = log_ps.shape
		log_ps = torch.reshape(log_ps, (batch_size*num_mics, frms, freq))
		tgt_mask = torch.reshape(tgt_mask, (batch_size*num_mics, frms, freq))

		_seq_len = torch.reshape(seq_len, (-1,))

		est_mask = self.forward(log_ps, _seq_len)
		loss = self.loss( est_mask, tgt_mask[:,:est_mask.shape[1],:], _seq_len)

		self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
		return {"loss" : loss , "est_mask" : est_mask }

	def validation_step(self, val_batch, batch_idx):
		log_ps, tgt_mask, seq_len, _, _ = val_batch

		batch_size, num_mics, frms, freq = log_ps.shape
		log_ps = torch.reshape(log_ps, (batch_size*num_mics, frms, freq))
		tgt_mask = torch.reshape(tgt_mask, (batch_size*num_mics, frms, freq))

		_seq_len = torch.reshape(seq_len, (-1,))

		est_mask = self.forward(log_ps, _seq_len)
		loss = self.loss( est_mask, tgt_mask[:,:est_mask.shape[1],:], _seq_len)
		self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
		return { "val_loss" : loss, "est_mask" : est_mask }

	# Multi-channel test step
	def test_step_v1(self, test_batch, batch_idx):
		_log_ps, _tgt_mask, mix_cs, seq_len, _ = test_batch
		batch_size, num_ch, t_frms, freq = _log_ps.shape
		#print(f'{batch_size}, {num_ch}, {t_frms}, {freq}')
		est_masks = torch.zeros(_tgt_mask.shape).type_as(_tgt_mask)
		get_single_ch = lambda x, ch: x[:,ch,:,:]
		for _ch in range(num_ch):
			est_mask = self.causal_forward(get_single_ch(_log_ps, _ch), seq_len)
			tgt_mask = get_single_ch(_tgt_mask, _ch)
			#print(f'{est_mask.shape}, {tgt_mask.shape}')
			test_loss = F.mse_loss(est_mask, tgt_mask)
			self.log('test_loss_mask_level', test_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

			est_masks[:,_ch,:,:] = est_mask
		
		return { "test_loss" : test_loss, "est_masks" : est_masks }

	# Multi-channel test step
	def test_step(self, test_batch, batch_idx):
		_log_ps, _tgt_mask, seq_len, doa, mix_cs = test_batch
		batch_size, num_ch, t_frms, freq = _log_ps.shape
		seq_len = seq_len.squeeze()
		#merging first two dimensions
		#assuming padded input

		_log_ps_rsh = torch.reshape(_log_ps, (batch_size*num_ch, t_frms, freq)) 
		_tgt_mask_rsh = torch.reshape(_tgt_mask, (batch_size*num_ch, t_frms, freq)) 
		#seq_len = torch.repeat_interleave(seq_len, 2)
		
		_est_masks = self.forward(_log_ps_rsh, seq_len)
		
		test_loss = self.loss(_est_masks, _tgt_mask_rsh, seq_len)
		self.log('test_loss_mask_level', test_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

		#est_masks = torch.reshape(_est_masks, (batch_size, num_ch, t_frms, freq)) 
		
		return { "test_loss" : test_loss, "est_masks" : _est_masks }


def main(args):
	#dbg_print(f"{torch.cuda.is_available()}, {torch.cuda.device_count()}\n")

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
			elif lst[0]=="inp_mean_var_norm_flag":
				inp_mean_var_norm_flag = True if lst[1]=="True" else False
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
	
	array_config['array_setup'] = get_array_set_up_from_config(array_config['array_type'], array_config['num_mics'], array_config['intermic_dist'])

	train_dataset = MovingSourceDataset(dataset_file, array_config, transforms=[ ZQ_NetworkInput(320, 160, ref_mic_idx, array_config['array_type'])], 
										T60=T60, SNR=SNR, dataset_dtype=dataset_dtype, dataset_condition=dataset_condition, train_flag=args.train,
										noise_simulation=noise_simulation, diffuse_files_path=diffuse_files_path) #, size=20)

	dev_dataset = MovingSourceDataset(val_dataset_file, array_config, transforms=[ ZQ_NetworkInput(320, 160, ref_mic_idx, array_config['array_type'])],
									T60=T60, SNR=SNR, dataset_dtype=dataset_dtype, dataset_condition=dataset_condition, train_flag=args.train,
									noise_simulation=noise_simulation, diffuse_files_path=diffuse_files_path) #, size=20



	# model
	bidirectional = args.bidirectional
	model = ZQRNN(bidirectional, train_dataset, dev_dataset, args.batch_size, args.num_workers, loss_flag)

	## exp path directories

	ckpt_dir = f'{args.ckpt_dir}/{loss_flag}/{dataset_dtype}/{dataset_condition}/{noise_simulation}/ref_mic_{ref_mic_idx}' #/input_mvn_norm_{inp_mean_var_norm_flag}'
	exp_name = f'{args.exp_name}' 

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
	precision=32
	trainer = pl.Trainer(accelerator='gpu', devices=args.num_gpu_per_node, num_nodes=args.num_nodes, precision=precision,
					max_epochs = args.max_n_epochs,
					callbacks=[checkpoint_callback, model_summary, lr_monitor],# pesq_checkpoint_callback, stoi_checkpoint_callback, DOAcallbacks()], early_stopping, GradNormCallback()
					logger=tb_logger,
					strategy="ddp_find_unused_parameters_false",
					check_val_every_n_epoch=1,
					log_every_n_steps = 1,
					num_sanity_val_steps=-1,
					profiler="simple",
					fast_dev_run=False,
					auto_scale_batch_size=False,
					detect_anomaly=True#,
					#gradient_clip_val=0.5
					)
	
	#trainer.tune(model)
	#print(f'Max batch size fit on memory: {model.batch_size}\n')
				
	msg = f"Train Config: bidirectional: {bidirectional}, loss_flag: {loss_flag}, precision: {precision}, \n \
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

	if 0:
		T60 = args.T60 
		SNR = args.SNR
		
	else:
		#reading from file for array jobs
		with open(args.input_test_filename, 'r') as f:
			lines = [line for line in f.readlines()]

		#print(lines)
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
				array_config["real_rirs"] = True if lst[1]=="True" else False
			elif lst[0]=="dist":
				array_config["dist"] = int(lst[1])
			elif lst[0]=="loss":
				loss_flag = lst[1]
			elif lst[0]=="inp_mean_var_norm_flag":
				inp_mean_var_norm_flag = True if lst[1]=="True" else False
			elif lst[0]=="model_path":
				model_path = lst[1]
			else:
				continue
	
	dataset_condition = args.dataset_condition
	dataset_dtype = args.dataset_dtype
	noise_simulation = args.noise_simulation
	diffuse_files_path = args.diffuse_files_path
	array_config['array_setup'] = get_array_set_up_from_config(array_config['array_type'], array_config['num_mics'], array_config['intermic_dist'])
	
	num_mics = array_config["num_mics"]
	mic_pairs = [(mic_1, mic_2) for mic_1 in range(0, num_mics) for mic_2 in range(mic_1+1, num_mics)]

	test_dataset = MovingSourceDataset(dataset_file, array_config, transforms=[ ZQ_NetworkInput(320, 160, ref_mic_idx, array_config['array_type'])], 
										T60=T60, SNR=SNR, dataset_dtype=dataset_dtype, dataset_condition=dataset_condition, train_flag=args.train,
										noise_simulation=noise_simulation, diffuse_files_path=diffuse_files_path) #, size=15)
	test_loader = DataLoader(test_dataset, batch_size = args.batch_size,
							 num_workers=args.num_workers, pin_memory=True, drop_last=True)  

	## exp path directories

	#ckpt_dir = f'{args.ckpt_dir}/{dataset_dtype}/{dataset_condition}/ref_mic_{ref_mic_idx}'
	ckpt_dir = f'{args.ckpt_dir}/{loss_flag}/{dataset_dtype}/{dataset_condition}/{noise_simulation}/ref_mic_{ref_mic_idx}'  #{noise_simulation}

	if args.dataset_condition =="reverb":
		app_str = f't60_{T60}'
	elif args.dataset_condition =="noisy":
		app_str = f'snr_{SNR}dB'
	elif args.dataset_condition =="noisy_reverb":
		app_str = f't60_{T60}_snr_{SNR}dB'
	else:
		app_str = ''

	exp_name = f'Test_{args.exp_name}_{dataset_dtype}_{app_str}'
	
	tb_logger = pl_loggers.TensorBoardLogger(save_dir=ckpt_dir, version=exp_name)
	precision = 16
	trainer = pl.Trainer(accelerator='gpu', devices=args.num_gpu_per_node, num_nodes=args.num_nodes, precision=precision,
						callbacks=[ DOAcallbacks(array_config=array_config, doa_tol=doa_tol, doa_euclid_dist=doa_euclid_dist, mic_pairs=mic_pairs)], #Losscallbacks(),
						logger=tb_logger
						)
	bidirectional = args.bidirectional
	
	#getting model
	for _file in os.listdir(ckpt_dir):
		if _file.endswith(".ckpt") and _file[:5]=="epoch":
			model_path = _file

	msg = f"Test Config: bidirectional: {bidirectional}, batch_size: {args.batch_size}, precision: {precision}, \n \
		ckpt_dir: {ckpt_dir}, exp_name: {exp_name}, \n \
		model: {model_path}, ref_mic_idx : {ref_mic_idx}, \n \
		dataset_file: {dataset_file}, t60: {T60}, snr: {SNR}, dataset_dtype: {dataset_dtype}, dataset_condition: {dataset_condition} \n"

	trainer.logger.experiment.add_text("Exp details", msg)

	print(msg)

	if os.path.exists(os.path.join(ckpt_dir, model_path)):  # *****TODO args.model_path)):
		model = ZQRNN.load_from_checkpoint(os.path.join(ckpt_dir, model_path), bidirectional=bidirectional, 
												train_dataset=None, val_dataset=None, loss_flag=loss_flag)
		trainer.test(model, dataloaders=test_loader)
	else:
		print(f"Model path not found in {ckpt_dir}")


if __name__ =='__main__':

	args = parser.parse_args()
	if args.train:
		main(args)
	else:
		test(args)
	
