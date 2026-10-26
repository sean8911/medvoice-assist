#!/bin/bash
#SBATCH --job-name=yolo_dense
#SBATCH --output=logs/out_%j.txt
#SBATCH --error=logs/error_%j.txt
#SBATCH --gres=gpu:1

python train.py --model cfg/training/yolov8n_densenet121.yaml --name yolov8n_densenet121_baseline_640 --batch 8 --patience 0 --imgsz 640 --worker 2