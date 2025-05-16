#!/bin/bash
#SBATCH --job-name=YOLO10n
#SBATCH --output=/data/dxie/mot/drone-sort/output/01_train_yolo10n.txt
#SBATCH --partition=nukwa
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --nodelist=nukwa-05.cnca
bash /data/dxie/mot/drone-sort/bash/train_model.sh