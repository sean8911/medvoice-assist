#!/bin/bash
#SBATCH --job-name=thesis2_biobert
#SBATCH --output=logs/out_%j.txt
#SBATCH --error=logs/error_%j.txt
#SBATCH --gres=gpu:1

python train_biobert_and_compare.py --dataset-dir data/disease-diagnosis-dataset --baseline-dir baseline_outputs --output-dir outputs --epochs 1 --batch-size 16 --eval-batch-size 32 --gradient-accumulation-steps 1 --learning-rate 2e-5 --weight-decay 0.01 --max-length 128 --save-steps 1000 --resume-from-checkpoint auto
