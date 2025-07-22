import tensorflow as tf
import tensorflow_probability as tfp
from typing import Tuple, List
from config import *

class VAE_Encoder(tf.keras.Model):
    def __init__(self, 
                 seq_len: int = WINDOW_SIZE,
                 latent_dim: int = LATENT_DIM,
                 features: int = REDUCED_FEATURES,
                 lstm_units: List[int] = LSTM_UNITS):
        """
        Initialize the encoder
        
        Args:
            seq_len: Length of input sequence
            latent_dim: Dimension of latent space
            features: Number of input features
            lstm_units: List of units in BiLSTM layers
        """
        super(VAE_Encoder, self).__init__()
        self.seq_len = seq_len
        self.features = features
        self.latent_dim = latent_dim
        self.lstm_units = lstm_units
        self.encoder = self.build_BiLSTM_encoder()

    def build_BiLSTM_encoder(self) -> tf.keras.Model:
        """Build the encoder architecture"""
        # Input layer
        enc_input = tf.keras.layers.Input(shape=(self.seq_len, self.features))
        
        # Add small amount of Gaussian noise for regularization
        x = tf.keras.layers.GaussianNoise(0.01)(enc_input)
        
        # BiLSTM layers
        for units in self.lstm_units:
            x = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(units, return_sequences=True)
            )(x)
        
        # Output layers for distribution parameters
        z_mean = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(self.latent_dim, name="z_mean")
        )(x)
        
        z_log_var = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(self.latent_dim, name="z_log_var")
        )(x)
        
        # Reparameterization trick
        output_dist = tfp.distributions.Normal(loc=0., scale=1.)
        eps = output_dist.sample(tf.shape(z_mean))
        z = z_mean + tf.sqrt(tf.math.exp(z_log_var)) * eps
        
        return tf.keras.Model(
            enc_input, 
            [z_mean, z_log_var, z, x],  # x is the BiLSTM output for attention
            name="encoder"
        )

    @tf.function
    def call(self, inputs: tf.Tensor, **kwargs) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Forward pass"""
        return self.encoder(inputs, **kwargs) 