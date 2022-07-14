Generation of 12-Lead Electrocardiogram with Subject-Specific, Image-Derived Characteristics Using a Conditional Variational Autoencoder
-----

## Introduction
we propose a conditional variational autoencoder (cVAE) to automatically generate realistic 12-lead ECG signals. Generated ECGs can be adjusted to correspond to specific subject characteristics, particularly those from images. We demonstrate the ability of the model to adjust to age, sex and Body Mass Index (BMI) values.

## Network
![Network structure](https://ieeexplore.ieee.org/mediastore_new/IEEE/content/media/9761376/9761399/9761431/sang2-p5-sang-large.gif "Network Architecture")

## Development Setup
1. **Download the project code locally**
```
git clone https://github.com/YulingSang/CVAE_ECG
```
2. **Edit details in train.py**
(Edit the parameters and data directory, one can create a config.py file for easier input)

3. **Run train.py**
```
python train.py
```

## Main Files: Project Structure

  ```sh
  ├── README.md
  ├── vae.py *** the implementaion of conditional variational autoencoder
  ├── train.py ***  Runnable file
                    Edit the network structure and dataset input
  ├── vae_prediction.txt *** The VAE network which added an extra layer to the latent space for risk prediction
  ```
