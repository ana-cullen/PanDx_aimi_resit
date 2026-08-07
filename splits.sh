#!/bin/bash
#SBATCH --job-name=splits
#SBATCH --account=cseduimc037
#SBATCH --partition=csedu
#SBATCH --qos=csedu-normal
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=30G
#SBATCH --time=12:00:00
#SBATCH --output=splits_%j.out
#SBATCH --error=splits_%j.err

SCRIPT_DIR="/vol/csedu-nobackup/course/IMC037_aimi/group09/resit"

source "/vol/csedu-nobackup/course/IMC037_aimi/group09/resit/venv2/bin/activate"

export nnUNet_raw="$SCRIPT_DIR/nnUNet_raw"
export nnUNet_preprocessed="$SCRIPT_DIR/nnUNet_preprocessed"
export nnUNet_results="$SCRIPT_DIR/nnUNet_results"

python make_dase_splits.py

