# A Fast and Stable GAN for Small and High Resolution Imagesets - pytorch
The official pytorch implementation of the paper "Towards Faster and Stabilized GAN Training for High-fidelity Few-shot Image Synthesis", the paper can be found [here](https://arxiv.org/abs/2101.04775).

## 0. Data
The datasets used in the paper can be found at [link](https://drive.google.com/file/d/1aAJCZbXNHyraJ6Mi13dSbe7pTyfPXha0/view?usp=sharing). 

After testing on over 20 datasets with each has less than 100 images, this GAN converges on 80% of them.
I still cannot summarize an obvious pattern of the "good properties" for a dataset which this GAN can converge on, please feel free to try with your own datasets.
 

## 1. Description
The code is structured as follows:
* models.py: all the models' structure definition.

* operation.py: the helper functions and data loading methods during training.

* train.py: the main entry of the code, execute this file to train the model, the intermediate results and checkpoints will be automatically saved periodically into a folder "train_results".

* eval.py: generates images from a trained generator into a folder, which can be used to calculate FID score.

* benchmarking: the functions we used to compute FID are located here, it automatically downloads the pytorch official inception model. 

* lpips: this folder contains the code to compute the LPIPS score, the inception model is also automatically download from official location.

* scripts: this folder contains many scripts you can use to play around the trained model. Including: 
    1. style_mix.py: style-mixing as introduced in the paper;
    2. generate_video.py: generating a continuous video from the interpolation of generated images;
    3. find_nearest_neighbor.py: given a generated image, find the closest real-image from the training set;
    4. train_backtracking_one.py: given a real-image, find the latent vector of this image from a trained Generator.

## 2. How to run
Place all your training images in a folder, and simply call
```
python train.py --path /path/to/RGB-image-folder
```
You can also see all the training options by:
```
python train.py --help
```
The code will automatically create a new folder (you have to specify the name of the folder using --name option) to store the trained checkpoints and intermediate synthesis results.

Once finish training, you can generate 100 images (or as many as you want) by:
```
cd ./train_results/name_of_your_training/
python eval.py --n_sample 100 
```

## 3. Pre-trained models
The pre-trained models and the respective code of each model are shared [here](https://drive.google.com/drive/folders/1nCpr84nKkrs9-aVMET5h8gqFbUYJRPLR?usp=sharing).

You can also use FastGAN to generate images with a pre-packaged Docker image, hosted on the Replicate registry: https://beta.replicate.ai/odegeasslbc/FastGAN

## 4. Important notes
1. The provided code is for research use only.
2. Different model and training configurations are needed on different datasets. You may have to tune the hyper-parameters to get the best results on your own datasets. 

    2.1. The hyper-parameters includes: the augmentation options, the model depth (how many layers), the model width (channel numbers of each layer). To change these, you have to change the code in models.py and train.py directly. 
    
    2.2. Please check the code in the shared pre-trained models on how each of them are configured differently on different datasets. Especially, compare the models.py for ffhq and art datasets, you will get an idea on what chages could be made on different datasets.

## 5. Other notes
1. The provided scripts are not well organized, contributions are welcomed to clean them.
2.  An third-party implementation of this paper can be found [here](https://github.com/lucidrains/lightweight-gan), where some other techniques are included. I suggest you try both implementation if you find one of them does not work. 
