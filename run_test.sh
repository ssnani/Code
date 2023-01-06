#!/bin/bash
export CUDA_VISIBLE_DEVICES='0,1,2,3'
srun ~/.conda/envs/moving_source/bin/python train.py --num_nodes=$1 \
                  --num_gpu_per_node=$2 \
                  --T60=0.0 \
                  --SNR=-5.0 \
                  --dataset_dtype=stationary \
                  --dataset_condition=noisy \
                  --ref_mic_idx=1 \
                  --dataset_file=../test_dataset_file_circular_motion_snr_-5_t60_0.2.txt \
                  --val_dataset_file=../test_dataset_file_circular_motion_snr_-5_t60_0.2.txt \
                  --bidirectional \
                  --batch_size=1 \
                  --num_workers=1 \
                  --ckpt_dir=/scratch/bbje/battula12/ControlledExp/random_seg/stationary \
                  --exp_name=CircularMotion_Static \
                  --model_path=epoch=94-step=4275.ckpt

