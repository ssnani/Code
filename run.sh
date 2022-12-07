#!/bin/bash
export CUDA_VISIBLE_DEVICES='0,1,2,3'
srun ~/.conda/envs/moving_source/bin/python train.py --num_nodes=$1 \
                  --num_gpu_per_node=$2 \
                  --batch_size=128 \
                  --bidirectional \
                  --max_n_epochs=100 \
                  --ckpt_dir=/scratch/bbje/battula12/ControlledExp/MovingSrc/DCCRN/ref_mic_0 \
                  --ref_mic_idx=0 \
                  --exp_name=CircularMotion_-5db_snr_0.2_t60_mic_0 \
                  --train \
                  --dataset_file=../dataset_file_circular_motion_snr_-5_t60_0.2.txt \
                  --val_dataset_file=../val_dataset_file_circular_motion_snr_-5_t60_0.2.txt \
                  

                  

