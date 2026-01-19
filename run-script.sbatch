#!/bin/bash
#SBATCH --job-name=pytorch_job
#SBATCH --output=ntp.log
#SBATCH --partition=gpu      
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128gb

# Activate the virtual environment with PyTorch (use the correct path to your project)
source $HOME/nlphw/my_project.env



# Run your PyTorch script
#python train_model.py
python run_ntp.py
