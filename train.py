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
from dataset import MovingSourceDataset, NetworkInput
from array_setup import array_setup_10cm_2mic
from networks import Net
from loss_criterion import LossFunction
from arg_parser import parser
from debug import dbg_print
from callbacks import Losscallbacks

class DCCRN_model(pl.LightningModule):
	def __init__(self, bidirectional, train_dataset: Dataset, val_dataset: Dataset, batch_size=32, num_workers=4):
		super().__init__()
		pl.seed_everything(77)

		self.model = Net(bidirectional)
		self.loss = LossFunction()

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
		_optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, amsgrad=True)
		#_lr_scheduler = torch.optim.lr_scheduler.StepLR(_optimizer, step_size=2, gamma=0.98)
		_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(_optimizer, factor=0.5, patience=5)
		return {"optimizer": _optimizer, "lr_scheduler": _lr_scheduler, "monitor": 'val_loss'}

	def forward(self, input_batch):
		mix_ri_spec, tgt_ri_spec, doa = input_batch
		est_ri_spec = self.model(mix_ri_spec)
		return est_ri_spec, tgt_ri_spec


	def training_step(self, train_batch, batch_idx):
		est_ri_spec, tgt_ri_spec = self.forward(train_batch)
		loss, loss_ri, loss_mag = self.loss(est_ri_spec, tgt_ri_spec)

		self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True )
		self.log('loss_ri', loss_ri, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True )
		self.log('loss_mag', loss_mag, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True )
		return {"loss" : loss , "loss_ri": loss_ri, "loss_mag": loss_mag, "est_ri_spec" : est_ri_spec }


	def validation_step(self, val_batch, batch_idx):
		est_ri_spec, tgt_ri_spec = self.forward(val_batch)
		loss, loss_ri, loss_mag  = self.loss(est_ri_spec, tgt_ri_spec)

		self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True )
		self.log('val_loss_ri', loss_ri, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True )
		self.log('val_loss_mag', loss_mag, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True )
		return {"loss" : loss , "loss_ri": loss_ri, "loss_mag": loss_mag, "est_ri_spec" : est_ri_spec }


	def test_step(self, test_batch, batch_idx):
		est_ri_spec, tgt_ri_spec = self.forward(test_batch)
		loss, loss_ri, loss_mag  = self.loss(est_ri_spec, tgt_ri_spec)

		self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True )
		self.log('test_loss_ri', loss_ri, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True )
		self.log('test_loss_mag', loss_mag, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True )
		return {"loss" : loss , "loss_ri": loss_ri, "loss_mag": loss_mag, "est_ri_spec" : est_ri_spec }



	def validation_epoch_end(self, validation_step_outputs):
		tensorboard = self.logger.experiment
		if (self.current_epoch %10==0):
			for batch_idx, batch_output in enumerate(validation_step_outputs):
				#if ((batch_idx+self.current_epoch)%4==0):
				idx = random.randint(0, self.batch_size-1)
				est_ri_spec = batch_output['est_ri_spec'][idx]
				est_ri_spec = est_ri_spec.to(torch.float32)
				_est_ri_spec = torch.permute(est_ri_spec,[2,1,0])  #(2,T,F) -> (F,T,2)
				est_sig = torch.istft(_est_ri_spec, 320,160,320,torch.hamming_window(320).type_as(_est_ri_spec))
				#rank_zero_only
				tensorboard.add_audio(f'est_{self.current_epoch}_{batch_idx}', est_sig/torch.max(torch.abs(est_sig)), sample_rate=16000)


def main(args):
	dbg_print(f"{torch.cuda.is_available()}, {torch.cuda.device_count()}\n")
	T = 4
	ref_mic_idx = args.ref_mic_idx
	T60 = args.T60 
	SNR = args.SNR
	dataset_dtype = args.dataset_dtype
	dataset_condition = args.dataset_condition

	#Loading datasets
	#scenario = args.scenario
	
	dataset_file = args.dataset_file #f'../dataset_file_circular_{scenario}_snr_{snr}_t60_{t60}.txt'
	train_dataset = MovingSourceDataset(dataset_file, array_setup_10cm_2mic, transforms=[ NetworkInput(320, 160, ref_mic_idx)], 
										T60=T60, SNR=SNR, dataset_dtype=dataset_dtype, dataset_condition=dataset_condition) #


	val_dataset_file = args.val_dataset_file #f'../dataset_file_circular_{scenario}_snr_{snr}_t60_{t60}.txt'
	dev_dataset = MovingSourceDataset(val_dataset_file, array_setup_10cm_2mic, transforms=[ NetworkInput(320, 160, ref_mic_idx)],
									T60=T60, SNR=SNR, dataset_dtype=dataset_dtype, dataset_condition=dataset_condition)


	# model
	bidirectional = args.bidirectional
	model = DCCRN_model(bidirectional, train_dataset, dev_dataset, args.batch_size, args.num_workers)


	## exp path directories

	ckpt_dir = f'{args.ckpt_dir}/{dataset_dtype}/{dataset_condition}/ref_mic_{ref_mic_idx}'
	exp_name = f'{args.exp_name}_t60_{T60}_snr_{SNR}dB'

	msg_pre_trained = None
	if (not os.path.exists(os.path.join(ckpt_dir,args.resume_model))) and len(args.pre_trained_ckpt_path)>0:
		msg_pre_trained = f"Loading ONLY model parameters from {args.pre_trained_ckpt_path}"
		print(msg_pre_trained)
		ckpt_point = torch.load(args.pre_trained_ckpt_path)
		model.load_state_dict(ckpt_point['state_dict'],strict=False) 

	tb_logger = pl_loggers.TensorBoardLogger(save_dir=ckpt_dir, version=exp_name)
	checkpoint_callback = ModelCheckpoint(dirpath=ckpt_dir, save_last = True, save_top_k=1, monitor='val_loss')

	pesq_checkpoint_callback = ModelCheckpoint(dirpath=ckpt_dir, filename='{epoch}-{VAL_PESQ_NB:.2f}--{val_loss:.2f}-{VAL_STOI:.2f}', save_last = True, save_top_k=1, monitor='VAL_PESQ_NB')
	pesq_checkpoint_callback.CHECKPOINT_NAME_LAST = "pesq-nb-last"
	stoi_checkpoint_callback = ModelCheckpoint(dirpath=ckpt_dir, filename='{epoch}-{VAL_STOI:.2f}-{val_loss:.2f}-{VAL_PESQ_NB:.2f}', save_last = True, save_top_k=1, monitor='VAL_STOI')
	stoi_checkpoint_callback.CHECKPOINT_NAME_LAST = "stoi-last"

	model_summary = ModelSummary(max_depth=1)
	early_stopping = EarlyStopping('val_loss', patience=5)
	lr_monitor = LearningRateMonitor(logging_interval='step')


	# training
	trainer = pl.Trainer(accelerator='gpu', devices=args.num_gpu_per_node, num_nodes=args.num_nodes, precision=16,
					max_epochs = args.max_n_epochs,
					callbacks=[checkpoint_callback, model_summary, lr_monitor, early_stopping], #pesq_checkpoint_callback, stoi_checkpoint_callback, Losscallbacks()
					logger=tb_logger,
					strategy="ddp_find_unused_parameters_false",
					check_val_every_n_epoch=1,
					log_every_n_steps = 1,
					num_sanity_val_steps=-1,
					profiler="simple",
					fast_dev_run=False,
					auto_scale_batch_size=False)
	
	#trainer.tune(model)
	#print(f'Max batch size fit on memory: {model.batch_size}\n')
				
	msg = f"Train Config: bidirectional: {bidirectional}, T: {T} \n, \
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
	T60 = args.T60 
	SNR = args.SNR
	dataset_dtype = args.dataset_dtype
	dataset_condition = args.dataset_condition

	test_dataset = MovingSourceDataset(dataset_file, array_setup_10cm_2mic,# size=1,
									transforms=[ NetworkInput(320, 160, ref_mic_idx)],
									T60=T60, SNR=SNR, dataset_dtype=dataset_dtype, dataset_condition=dataset_condition) #
	test_loader = DataLoader(test_dataset, batch_size = args.batch_size,
							 num_workers=args.num_workers, pin_memory=True, drop_last=True)  

	## exp path directories

	#ckpt_dir = f'{args.ckpt_dir}/{dataset_dtype}/{dataset_condition}/ref_mic_{ref_mic_idx}'
	ckpt_dir = f'{args.ckpt_dir}/{dataset_condition}/ref_mic_{ref_mic_idx}'
	exp_name = f'Test_{args.exp_name}_t60_{T60}_snr_{SNR}dB'
	
	tb_logger = pl_loggers.TensorBoardLogger(save_dir=ckpt_dir, version=exp_name)

	trainer = pl.Trainer(accelerator='gpu', devices=args.num_gpu_per_node, num_nodes=args.num_nodes, precision=16,
						callbacks=[Losscallbacks()],
						logger=tb_logger
						)
	bidirectional = args.bidirectional
	
	msg = f"Test Config: bidirectional: {bidirectional}, T: {T}, batch_size: {args.batch_size}, \n \
		ckpt_dir: {ckpt_dir}, exp_name: {exp_name}, \n \
		model: {args.model_path}, ref_mic_idx : {ref_mic_idx}, \n \
		dataset_file: {dataset_file}, t60: {T60}, snr: {SNR}, dataset_dtype: {dataset_dtype}, dataset_condition: {dataset_condition} \n"

	trainer.logger.experiment.add_text("Exp details", msg)

	print(msg)

	if os.path.exists(os.path.join(ckpt_dir, args.model_path)):
		model = DCCRN_model.load_from_checkpoint(os.path.join(ckpt_dir, args.model_path), bidirectional=bidirectional, 
		        								train_dataset=None, val_dataset=None)
		trainer.test(model, dataloaders=test_loader)
	else:
		print(f"Model path not found in {ckpt_dir}")

if __name__=="__main__":
	#flags
	args = parser.parse_args()
	if args.train:
		main(args)
	else:
		print("Testing\n")
		print(f"{torch.cuda.is_available()}, {torch.cuda.device_count()}\n")
		test(args)
