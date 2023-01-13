#!/bin/bash
export CUDA_VISIBLE_DEVICES='0,1,2,3'
srun ~/.conda/envs/moving_source/bin/python train.py --num_nodes=$1 \
                  --num_gpu_per_node=$2 \
                  --T60=0.0 \
                  --SNR=-5.0 \
                  --dataset_dtype=stationary \
                  --dataset_condition=noisy_reverb \
                  --ref_mic_idx=0 \
                  --train \
                  --dataset_file=../dataset_file_circular_motion_snr_-5_t60_0.2_noisy_reverb.txt \
                  --val_dataset_file=../val_dataset_file_circular_motion_snr_-5_t60_0.2_noisy_reverb.txt \
                  --bidirectional \
                  --batch_size=128 \
                  --max_n_epochs=100 \
                  --num_workers=4 \
                  --ckpt_dir=/scratch/bbje/battula12/ControlledExp/random_seg \
                  --exp_name=CircularMotion \
                  --resume_model=last.ckpt