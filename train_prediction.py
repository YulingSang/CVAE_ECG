import numpy as np
# import tensorflow as tf
from vae_prediction import VAE

LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 1


def train(x_train, y_train, c_train, x_val, y_val, c_val, learning_rate, batch_size, epochs):
    vae = VAE(
        input_shape=(12, 400, 1),
        conv_filters=(5, 5),
        conv_kernels=((1, 40), (12, 1)),
        conv_paddings=("same", "valid"),
        latent_space_dim=10,
        condition_dim=32
    )
    vae.summary()
    vae.compile(learning_rate)
    vae.train(x_train, y_train, c_train, x_val, y_val, c_val, batch_size, epochs)

    return vae

if __name__ == "__main__":
    X_tr = np.load("")
    X_tr = X_tr.reshape(X_tr.shape + (1,))
    Y_tr = np.load("")

    X_te = np.load("")
    X_te = X_te.reshape(X_te.shape + (1,))
    Y_te = np.load("")

    Con_tr = np.load("")
    Con_te = np.load("")

    vae = train(X_tr, Y_tr, Con_tr, X_te, Y_te, Con_te, LEARNING_RATE, BATCH_SIZE, EPOCHS)
    vae.save("model")

