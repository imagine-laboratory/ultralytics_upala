#!/bin/bash
#SBATCH --job-name=mosaic-drone
#SBATCH --output=/data/mvarela/datasets/test/corona-mosaic-dron-output/01_test_yolo.txt
#SBATCH --partition=nukwa
#SBATCH --nodelist=nukwa-02.cnca
#SBATCH --ntasks=1
#SBATCH --time=24:00:00


# Ejecutar el script de entrenamiento
bash /data/mvarela/ultralytics_upala/bash/train_model.sh