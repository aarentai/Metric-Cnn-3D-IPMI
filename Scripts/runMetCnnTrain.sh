#!/usr/bin/env bash

python MetricCnnTrainingInference.py --brain_id=111312 --input_dir=../Brains --output_dir=../Checkpoints --gpu_device=1 --epoch_num=2 --learning_rate=1e-4 --terminating_loss=1e6 --checkpoint_save_frequency=1 
