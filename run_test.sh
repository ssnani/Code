#!/bin/bash
export CUDA_VISIBLE_DEVICES='0,1,2,3'
srun ~/.conda/envs/moving_source/bin/python train.py --num_nodes=$1 \
                  --num_gpu_per_node=$2 \
                  --batch_size=1 \
                  --bidirectional \
                  --ckpt_dir=/scratch/bbje/battula12/ControlledExp/StationarySrc/DCCRN/ref_mic_1_with_ref_mic_0_pre_trained_init \
                  --ref_mic_idx=1 \
                  --exp_name=Val_CircularMotion_-5db_snr_0.2_t60_mic_1_transfer_learning \
                  --num_workers=1 \
                  --dataset_file=../val_dataset_file_circular_static_snr_-5_t60_0.2.txt \
                  --val_dataset_file=../val_dataset_file_circular_static_snr_-5_t60_0.2.txt \
                  --model_path=epoch=48-step=2205.ckpt

