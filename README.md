Generation of 12-Lead Electrocardiogram with Subject-Specific, Image-Derived Characteristics Using a Conditional Variational Autoencoder
-----

## Introduction
This repository is the source demo code for conference paper "Generation of 12-Lead Electrocardiogram with Subject-Specific, Image-Derived Characteristics Using a Conditional Variational Autoencoder". [Paperlink.](https://ieeexplore.ieee.org/abstract/document/9761431)


we propose a conditional variational autoencoder (cVAE) to automatically generate realistic 12-lead ECG signals. Generated ECGs can be adjusted to correspond to specific subject characteristics, particularly those from images. We demonstrate the ability of the model to adjust to age, sex and Body Mass Index (BMI) values.

## Network
![Network structure](https://ieeexplore.ieee.org/mediastore_new/IEEE/content/media/9761376/9761399/9761431/sang2-p5-sang-large.gif "Network Architecture")

## Development Setup
1. **Download the project code locally**
```
git clone https://github.com/YulingSang/CVAE_ECG
```
2. **Install tensorflow2.0+**


## Main Files: 

  ```sh
  ├── README.md
  ├── train.py ***  Runnable file
                    Edit the network structure and dataset input
  ├── vae.py *** the implementaion of conditional variational autoencoder
  ├── vae_prediction.py *** The VAE network which added an extra layer to the 
                            latent space to realize risk prediction
  ```
 
## Instructions
 
1.  Edit details in train.py
    * Edit the global parameters like epoch, batchsize, learning rate
    * Edit network structure, especially the input_shape and convolutional layers
    * Edit data directory 
    * Or one can create a config.py file for easier input
    
2.  Edit the loss function
    * If you want to use vae_prediction.py, make sure you edit the loss function in vae_prediction.py as you need
    * For this work, we use Cox proportional hazard regression model and use negative cox proportional hazards partial likelihood as the prediction loss
    * If you have tasks like classification, please change this into correct loss function, e.g. cross_entropy

3.  Just run the train.py
    ```
    python train.py
    ```
    * The model parameters and weights can be saved using save() function in VAE class.

  
