import keras
from keras import layers
import numpy as np

def create_windows(signal, window_size=50):
    """
    Slices a 1D signal into overlapping 2D matrices.
    
    Args:
        signal: 1D numpy array
        window_size: Size of the window (default 50)
        
    Returns:
        2D numpy array of shape (num_windows, window_size)
    """
    # Number of windows is len(signal) - window_size + 1
    # However, for simplicity and to match common patterns, we can use a step of 1
    # or a step that avoids too much overlap if needed. The task says "overlapping".
    num_windows = len(signal) - window_size + 1
    windows = np.zeros((num_windows, window_size))
    for i in range(num_windows):
        windows[i] = signal[i:i + window_size]
    return windows

class SignalCompression(layers.Layer):
    """
    Custom layer to compress a signal window into a latent representation.
    """
    def __init__(self, latent_dim=8, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim

    def build(self, input_shape):
        # input_shape is (batch, window_size)
        self.w = self.add_weight(
            shape=(input_shape[-1], self.latent_dim),
            initializer="glorot_uniform",
            trainable=True,
            name="kernel"
        )
        self.b = self.add_weight(
            shape=(self.latent_dim,),
            initializer="zeros",
            trainable=True,
            name="bias"
        )

    def call(self, inputs):
        # Forward pass: Dense transformation with ReLU
        return keras.activations.relu(keras.ops.matmul(inputs, self.w) + self.b)

class SignalExpansion(layers.Layer):
    """
    Custom layer to expand a latent representation back to the original window size.
    """
    def __init__(self, output_dim=50, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim

    def build(self, input_shape):
        # input_shape is (batch, latent_dim)
        self.w = self.add_weight(
            shape=(input_shape[-1], self.output_dim),
            initializer="glorot_uniform",
            trainable=True,
            name="kernel"
        )
        self.b = self.add_weight(
            shape=(self.output_dim,),
            initializer="zeros",
            trainable=True,
            name="bias"
        )

    def call(self, inputs):
        # Forward pass: Dense transformation (usually linear for reconstruction)
        return keras.ops.matmul(inputs, self.w) + self.b

class PhysicsAutoencoder(keras.Model):
    """
    Subclassed Model combining SignalCompression and SignalExpansion.
    """
    def __init__(self, window_size=50, latent_dim=8, **kwargs):
        super().__init__(**kwargs)
        self.encoder = SignalCompression(latent_dim=latent_dim)
        self.decoder = SignalExpansion(output_dim=window_size)

    def call(self, inputs):
        latent = self.encoder(inputs)
        reconstructed = self.decoder(latent)
        return reconstructed

if __name__ == "__main__":
    # Quick test
    test_signal = np.random.rand(1000)
    windows = create_windows(test_signal, 50)
    print(f"Windows shape: {windows.shape}")
    
    model = PhysicsAutoencoder(window_size=50, latent_dim=8)
    # Build the model by calling it with dummy data
    output = model(windows[:1])
    print(f"Output shape: {output.shape}")
    model.summary()
