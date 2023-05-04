#!/bin/bash

#SBATCH -p palamut-cuda      # Kuyruk adi: Uzerinde GPU olan kuyruk olmasina dikkat edin.
#SBATCH -A proj3
#SBATCH -J Gaugan_Fat        # Gonderilen isin ismi
#SBATCH -o gaugan_fat.log    # Ciktinin yazilacagi dosya adi
#SBATCH --gres=gpu:3       # Her bir sunucuda kac GPU istiyorsunuz? Kumeleri kontrol edin.
#SBATCH -N 1                # Gorev kac node'da calisacak?
#SBATCH -n 1                # Ayni gorevden kac adet calistirilacak?
#SBATCH --cpus-per-task 48  # Her bir gorev kac cekirdek kullanacak? Kumeleri kontrol edin.
#SBATCH --time=48:00:00      # Sure siniri koyun.


eval "$(/truba/home/isahin/miniconda3/bin/conda shell.bash hook)"
conda activate gaugan_pytorch

base_dir='/truba/home/isahin/gaugan_pytorch/gaugan_fat'


python "$base_dir/train.py" >fat_training.txt