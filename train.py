import numpy as np
# import tensorflow as tf
from vae import VAE

# Global setting for cVAE including learning rate, batch size and epochs
# One can use a config file to set these variables
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 300

# hyperparameter setting, including the input shape, filter dimensions,
# convolutional kernel size, padding mode, latent space dimensions and conditions dimension.
def train(x_train, c_train, x_val, c_val, learning_rate, batch_size, epochs):
    vae = VAE(

        # The shape of ECG input. If your dataset is 8 leads and each lead has 500 points, please
        # use (8, 500, 1) as the input shape
        input_shape=(12, 400, 1),

        # The filter dimension is suitable for UK Biobank ECG median data, please select the size that
        # gives you best reconstruction results
        conv_filters=(5, 5),

        # In our model design, we have one horizontal conv kernel and one vertical conv kernel
        # first kernel is set to (1, x) where x depends on your ECG size
        # second kernel is set to (x, 1) where x is equal to number of leads
        conv_kernels=((1, 40), (12, 1)),

        # padding mode of each kernel
        conv_paddings=("same", "valid"),

        # latent space dimension, 10 dimension has good results on UK Biobank data
        # One can adjust it to smaller or larger depending on the data size
        latent_space_dim=10,

        # The dimension of conditions. If your condition is a True-or-False variable, you can keep
        # it as 1. If more conditions, better suggestion is to use one-hot coding.
        condition_dim=1
    )
    vae.summary()
    vae.compile(learning_rate)
    vae.train(x_train, c_train, x_val, c_val, batch_size, epochs)

    return vae

if __name__ == "__main__":

    # load the training and validation data
    # can also keep the route to directory into a config file
    X_tr = np.load("")
    X_tr = X_tr.reshape(X_tr.shape + (1,)) # to align with our input shape
    X_te = np.load("")
    X_te = X_te.reshape(X_te.shape + (1,))
    Con_tr = np.load("")
    Con_te = np.load("")

    vae = train(X_tr, Con_tr, X_te, Con_te, LEARNING_RATE, BATCH_SIZE, EPOCHS)

    # save functions will save a "parameters.pkl" and a "weights.h5" file in folder "model"
    # One can use vae.load("model") to load the weights and parameters later for test
    vae.save("model")

