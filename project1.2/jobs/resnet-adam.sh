#!/bin/sh
#BSUB -q gpua100
#BSUB -J hotdog-adam
#BSUB -n 8
#BSUB -R "span[block=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 1:30
#BSUB -R "rusage[mem=32GB]"
#BSUB -u andersbthuesen@gmail.com
#BSUB -B
#BSUB -N
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err

nvidia-smi
module load cuda/11.1.1

PATH=~/miniconda3/bin:$PATH

python main.py --model ResNet --optimizer Adam --lr 0.001 --epochs 1 --augmentation True