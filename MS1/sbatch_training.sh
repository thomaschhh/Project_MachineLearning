$ touch sbatch_training.sh # create empty file
# Code for sbatch skript:

#!/bin/bash
#
#SBATCH --job-name=train # name appaering in squeue
#SBATCH --output=train_out.txt # terminal output is saved here
#SBATCH -p gpu --gres=gpu:tesla:1 # last number defines no of cores
date
hostname
python ./train.py
date
