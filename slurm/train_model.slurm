#!/bin/bash
#SBATCH --job-name=testYOLO
#SBATCH --output=/data/mvarela/mot/test/output/01_test_yolo.txt
#SBATCH --partition=nukwa
#SBATCH --nodelist=nukwa-07.cnca
#SBATCH --ntasks=1
#SBATCH --time=24:00:00


# Ejecutar el script de entrenamiento
bash /data/mvarela/pineapple-model/bash/train_model.sh