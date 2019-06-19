#!/bin/sh 
#SBATCH --ntasks=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --mail-user=xianlong.zeng@nationwidechildrens.org
#SBATCH --mail-type=ALL
python3 -u train.py
