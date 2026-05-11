import keras
from keras import layers
import numpy as np

class ConvSignalCompression(layers.Layer):
    """
    Refactored encoder using Conv1D to capture local temporal patterns.
    """
    def __init__(self, latent_dim=8, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        # Convolutional layers expect (batch, steps, channels)
        self.conv1 = layers.Conv1D(filters=16, kernel_size=3, activation='relu', padding='same')
        self.pool1 = layers.MaxPooling1D(pool_size=2)
        self.conv2 = layers.Conv1D(filters=latent_dim, kernel_size=3, activation='relu', padding='same')
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(latent_dim, activation='relu')

    def call(self, inputs):
        # inputs shape: (batch, 50) -> need to add channel dim: (batch, 50, 1)
        x = keras.ops.expand_dims(inputs, axis=-1)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        return self.dense(x)

class ConvSignalExpansion(layers.Layer):
    """
    Refactored decoder using Conv1DTranspose for reconstruction.
    """
    def __init__(self, output_dim=50, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        # Start from latent dim and upsample
        self.dense = layers.Dense(25 * 8, activation='relu') # 25 * latent_filters
        self.reshape = layers.Reshape((25, 8))
        self.conv_t = layers.Conv1DTranspose(filters=1, kernel_size=3, activation='linear', padding='same', strides=2)

    def call(self, inputs):
        x = self.dense(inputs)
        x = self.reshape(x)
        x = self.conv_t(x)
        # Reshape back to (batch, 50)
        return keras.ops.squeeze(x, axis=-1)

class ConvPhysicsAutoencoder(keras.Model):
    """
    Subclassed Model using Convolutional layers.
    """
    def __init__(self, window_size=50, latent_dim=8, **kwargs):
        super().__init__(**kwargs)
        self.encoder = ConvSignalCompression(latent_dim=latent_dim)
        self.decoder = ConvSignalExpansion(output_dim=window_size)

    def call(self, inputs):
        latent = self.encoder(inputs)
        reconstructed = self.decoder(latent)
        return reconstructed

# Explanation for Exercise 3:
# Conv1D layers are mathematically superior for local patterns in time series because they utilize 
# weight sharing and local connectivity. This allows the model to learn filters that detect 
# specific features (like the harmonic peaks in our RC-filter signal) regardless of where they 
# occur in the window, significantly improving the robustness against noise and shifts compared 
# to a static dense matrix.
