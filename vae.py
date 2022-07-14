import os
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv2D, ELU, BatchNormalization, \
    Flatten, Dense, Reshape, Conv2DTranspose, Lambda, AveragePooling2D, \
    UpSampling2D, Concatenate, AveragePooling1D
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Adam

tf.compat.v1.disable_eager_execution()


class MyCallBack(Callback):
    def __init__(self, loss_weight):
        super().__init__()
        self.loss_weight = loss_weight

    def on_epoch_begin(self, epoch, logs=None):
        if epoch == 0:
            K.set_value(self.loss_weight["encoder"], 1)
            K.set_value(self.loss_weight["decoder"], 1)


class VAE:
    def __init__(self,
                 input_shape,
                 conv_filters,
                 conv_kernels,
                 conv_paddings,
                 latent_space_dim,
                 condition_dim):
        self.input_shape = input_shape  # [15, 400, 1]
        self.conv_filters = conv_filters  # [5, 5]
        self.conv_kernels = conv_kernels  # [(1,40), (15,1)]
        self.conv_paddings = conv_paddings  # ["same", "valid"]
        self.latent_space_dim = latent_space_dim
        self.condition_dim = condition_dim
        # self.reconstruction_loss_weight = 1

        self.encoder = None
        self.decoder = None
        self.classifier = None
        self.model = None

        self._num_conv_layers = len(conv_filters)
        self._shape_before_bottleneck = None
        self._model_input = None
        self._model_condition = None

        self.loss_weight = {
            "encoder": K.variable(1),
            "decoder": K.variable(1)
        }

        self.trainingHistory = None

        self._build()

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

    def compile(self, learning_rate=0.0001):
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer,
                           loss={
                               "encoder": self._calculate_kl_loss,
                               "decoder": self._calculate_reconstruction_loss
                           },
                           loss_weights=self.loss_weight,
                           )

    def _calculate_reconstruction_loss(self, y_target, y_predicted):
        error = y_target - y_predicted
        reconstruction_loss = K.mean(K.square(error), axis=[1, 2, 3])
        return reconstruction_loss

    def _calculate_kl_loss(self, y_target, y_predicted):
        kl_loss = -0.5 * K.sum(1 + self.log_variance - K.square(self.mu) - K.exp(self.log_variance), axis=1)
        return kl_loss

    def _calculate_classification_loss(self, E, risk):
        hazard_ratio = K.exp(risk)
        log_risk = K.log(K.cumsum(hazard_ratio))
        uncensored_likelihood = risk - log_risk
        censored_likelihood = uncensored_likelihood * E
        neg_likelihood = -K.sum(censored_likelihood)
        return neg_likelihood

    def train(self, x_tr, c_tr, x_val, c_val, batch_size, num_epochs):
        history = self.model.fit([x_tr, c_tr],
                                 {
                                     "encoder": x_tr,
                                     "decoder": x_tr
                                 },
                                 batch_size=batch_size,
                                 epochs=num_epochs,
                                 validation_data=([x_val, c_val], [x_val, x_val]),
                                 callbacks=[MyCallBack(self.loss_weight)],
                                 shuffle=False)
        self.trainingHistory = history

    def save(self, save_folder="."):
        self._create_save_folder_if_it_doesnt_exsit(save_folder)
        self._save_parameters(save_folder)
        self._save_weights(save_folder)

    def _create_save_folder_if_it_doesnt_exsit(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

    def _save_parameters(self, save_folder):
        parameters = [
            self.input_shape,
            self.conv_filters,
            self.conv_kernels,
            self.conv_paddings,
            self.latent_space_dim,
            self.condition_dim
        ]
        save_path = os.path.join(save_folder, "parameters.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(parameters, f)

    def _save_weights(self, save_folder):
        save_path = os.path.join(save_folder, "weights.h5")
        self.model.save_weights(save_path)

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    @classmethod
    def load(cls, save_folder="."):
        parameters_path = os.path.join(save_folder, "parameters.pkl")
        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)
        vae = VAE(*parameters)
        weights_path = os.path.join(save_folder, "weights.h5")
        vae.load_weights(weights_path)

        return vae

    def _build(self):
        self._build_encoder()
        self._build_decoder()
        # self._build_classifier()
        self._build_autoencoder()

    def _build_encoder(self):
        encoder_input = self._add_encoder_input()
        condition = self._add_condition_input()
        conv_layers = self._add_conv_layers(encoder_input)
        bottleneck = self._add_bottleneck(conv_layers, condition)
        self._model_input = encoder_input
        self._model_condition = condition
        self.encoder = Model([encoder_input, condition], bottleneck, name="encoder")

    def _add_encoder_input(self):
        return Input(shape=self.input_shape, name="encoder_input")

    def _add_condition_input(self):
        return Input(shape=self.condition_dim, name="condition_input")

    def _add_conv_layers(self, encoder_input):
        """ Creates all convolutional blocks in encoder """
        x = encoder_input
        for layer_index in range(self._num_conv_layers):
            x = self._add_conv_layer(layer_index, x)
        return x

    def _add_conv_layer(self, layer_index, x):
        """ Adds a convolutional block to a graph of layers, consisting of conv2d
    + ReLu + batch normalization"""

        layer_number = layer_index + 1

        conv_layer = Conv2D(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            padding=self.conv_paddings[layer_index],
            name=f"encoder_conv_layer_{layer_number}"
        )

        x = conv_layer(x)
        x = ELU(name=f"encoder_relu_{layer_number}")(x)
        x = BatchNormalization(name=f"encoder_bn_{layer_number}")(x)

        return x

    def _add_bottleneck(self, x, condition):
        """Mean Sampling"""
        x = AveragePooling2D(pool_size=(1, 2))(x)

        """Flatten data and add bottleneck with Gaussian Sampling(Dense Layer). """
        self._shape_before_bottleneck = K.int_shape(x)[1:]  # [2 (batch size), 7, 7, 32]
        x = Flatten()(x)

        x = Concatenate(axis=-1)([x, condition])
        x = BatchNormalization()(x)

        self.mu = Dense(self.latent_space_dim, name="mu")(x)
        self.log_variance = Dense(self.latent_space_dim, name="log_variance")(x)

        def sample_point_from_normal_distribution(args):
            mu, log_variance = args
            epsilon = K.random_normal(shape=K.shape(self.mu), mean=0., stddev=1.)
            sampled_point = mu + K.exp(log_variance / 2) * epsilon
            return sampled_point

        x = Lambda(sample_point_from_normal_distribution, name="encoder_output")([self.mu, self.log_variance])

        return x

    def _build_decoder(self):
        decoder_input = self._add_decoder_input()
        condition = self._add_condition_input()
        con_decoder_input = self._add_con_decoder_input(decoder_input, condition)
        dense_layer = self._add_dense_layer(con_decoder_input)
        reshape_layer = self._add_reshape_layer(dense_layer)
        conv_transpose_layers = self._add_conv_transpose_layers(reshape_layer)
        decoder_output = self._add_decoder_output(conv_transpose_layers)
        self.decoder = Model([decoder_input, condition], decoder_output, name="decoder")

    def _add_decoder_input(self):
        return Input(shape=self.latent_space_dim, name="decoder_input")

    def _add_con_decoder_input(self, x, condition):
        return Concatenate(axis=-1)([x, condition])

    def _add_dense_layer(self, decoder_input):
        dense_layer = BatchNormalization()(decoder_input)
        num_neurons = np.prod(self._shape_before_bottleneck)  # [4, 4, 32] -> 4*4*32
        dense_layer = Dense(num_neurons, name="decoder_dense")(dense_layer)
        return dense_layer

    def _add_reshape_layer(self, dense_layer):
        reshape_layer = Reshape(self._shape_before_bottleneck)(dense_layer)
        reshape_layer = UpSampling2D(size=(1, 2))(reshape_layer)
        return reshape_layer

    def _add_conv_transpose_layers(self, x):
        """Add conv transpose blocks."""
        # loop through all the conv layers in reverse order and stop at the first layer
        for layer_index in reversed(range(1, self._num_conv_layers)):  # [0, 1, 2] -> [2, 1]
            x = self._add_conv_transpose_layer(layer_index, x)

        return x

    def _add_conv_transpose_layer(self, layer_index, x):
        layer_number = self._num_conv_layers - layer_index

        conv_transpose_layer = Conv2DTranspose(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            padding=self.conv_paddings[layer_index],
            name=f"decoder_conv_transpose_layer_{layer_number}"
        )

        x = conv_transpose_layer(x)
        x = ELU(name=f"decoder_relu_{layer_number}")(x)
        x = BatchNormalization(name=f"decoder_bn_{layer_number}")(x)

        return x

    def _add_decoder_output(self, x):
        conv_transpose_layer = Conv2DTranspose(
            filters=1,
            kernel_size=self.conv_kernels[0],
            padding=self.conv_paddings[0],
            name=f"decoder_conv_transpose_layer_{self._num_conv_layers}"
        )
        x = conv_transpose_layer(x)
        x = ELU(name=f"decoder_relu_{self._num_conv_layers}")(x)
        output_layer = BatchNormalization(name=f"decoder_bn_{self._num_conv_layers}")(x)
        return output_layer

    def _build_autoencoder(self):
        model_input = self._model_input
        condition = self._model_condition
        model_encoder_output = self.encoder([model_input, condition])
        model_decoder_output = self.decoder([model_encoder_output, condition])
        self.model = Model([model_input,condition] , [model_encoder_output, model_decoder_output],
                           name="VAE")
