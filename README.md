# A Robust Image Synthesis and Segmentation Pipeline For Histopathology

<img src="imgs/Figure 1.jpg" width="800px"/>

### [Paper](https://www.nature.com/articles/s41551-022-00952-9) | [Micro/Macrovesicular Steatosis of liver (MSL)](https://portal.gdc.cancer.gov/projects/TCGA-GBM) | [Breast Carcinoma Tubules (BCT)](https://portal.gdc.cancer.gov/projects/TCGA-LGG) | [Prostate Carcinoma Glands (PCG)](https://portal.gdc.cancer.gov/projects/TCGA-LUAD) 

In this paper, we propose a novel GAN-based image synthesis framework, PathopixGAN, specifically designed to offer more precise semantic control over the content and structure of the synthesized images. Besides, we proposed semantic segmentation methodology, PathoSeg, and show the superior performance of the method both quantitatively and qualitatively compared to the other high-performing SOTA models when trained on both real and real+PathopixGAN-generated synthetic data.

<br>
<img src='imgs/Figure 2.jpg' align="right" width=960>
<br>

## Example Results

### PathopixGAN synthetic data generation
<img src="imgs/Figure 3.jpg" width="800px"/>

### Qualitative comparison of Original and PathopixGAN synthetic data
<img src="imgs/Figure 4.jpg" width="800px"/>


## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN


### Getting started

- Clone this repo:
```bash
git clone https://github.com/DeepMIALab/PathSeg-PathopixGAN.git
cd PathSeg-PathopixGAN
```

- Install PyTorch and other dependencies with the environment.yml or requirements.txt.

- For pip users, please type the command `pip install -r requirements.txt`.

- For Conda users,  you can create a new Conda environment using `conda env create -f environment.yml`.

### Training and Test

- The training and test cohorts have been already provided with the dataset. You may download the dataset and operate the dataloader function provided in [PathSeg-PathopixGAN/dataloader](https://github.com/DeepMIALab/PathSeg-PathopixGAN/dataloader)  to replicate the results published in the paper. 

The data used for training are expected to be organized as follows:
```bash
Data_Path                # DIR_TO_TRAIN_DATASET
 ├──  img_dir
 |      ├── train
 |      |     ├── 1.png     
 |      |     ├── ...     
 |      |     └── m.png          
 |      ├── test
 |      |     ├── 1.png     
 |      |     ├── ...     
 |      |     └── m.png  
 |      └── validation
 |           ├── 1.png     
 |           ├── ...     
 |           └── m.png  
 ├──  ann_dir
 |      ├── train
 |      |     ├── 1.png     
 |      |     ├── ...     
 |      |     └── m.png          
 |      ├── test
 |      |     ├── 1.png     
 |      |     ├── ...     
 |      |     └── m.png  
 |      └── validation
 |           ├── 1.png     
 |           ├── ...     
 |           └── m.png  

```

- Train the MSL/BCT segmentation model:
```bash
python train.py --data_path_ ./dataset --encoder tu-hrnet_w30 --encoder_weights imagenet --train_batchsize 64 --validation_batchsize 32 --learning_rate 1e-4 --num_classes 2 --max_epochs 700 --gpus 3
```
```

- Train the PCG segmentation model:
```bash
python train.py --data_path_ ./dataset --encoder tu-hrnet_w30 --encoder_weights imagenet --train_batchsize 64 --validation_batchsize 32 --learning_rate 1e-4 --num_classes 5 --max_epochs 700 --gpus 3
```
```

- Test the PathoSeg model:
```bash
python test.py --datapath ./test_path  --inference_path ./inference_masks --ckpt checkpoints/best_model.pt --dataset_type Fat 
```

The test results for a dataset e.g Prostate will be saved here: ```./inference_masks/Prostate/` 


## Reference

If you find our work useful in your research or if you use parts of this code please consider citing our paper:

```



### Acknowledgments
Our PathopixGAN code is developed based on [SPADE](https://github.com/NVlabs/SPADE). We also thank [FastGan-pytorch](https://github.com/odegeasslbc/FastGAN-pytorch) for their pipeline for synthesizing semantic masks for PathopixGAN. 