#!/bin/sh
#BSUB -q gpua100
#BSUB -J 02-baseline-adam-no-aug
#BSUB -n 8
#BSUB -R "span[block=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 1:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -u joachimes@gmail.com
#BSUB -B
#BSUB -N
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err

nvidia-smi
module load cuda/11.1.1

PATH=~/miniconda3/bin:$PATH

python main.py --epochs 50 --batch-size 128 --loss "BCE"
