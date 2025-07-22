import tensorflow as tf
import tensorflow_probability as tfp
from typing import Tuple, List
from config import *

class VAE_Decoder(tf.keras.Model):
    def __init__(self,
                 seq_len: int = WINDOW_SIZE,
                 latent_dim: int = LATENT_DIM,
                 features: int = REDUCED_FEATURES,
                 lstm_units: List[int] = LSTM_UNITS[::-1]):  # Reverse LSTM units for decoder
        """
        Initialize the decoder
        
        Args:
            seq_len: Length of input sequence
            latent_dim: Dimension of latent space
            features: Number of output features
            lstm_units: List of units in BiLSTM layers (reversed from encoder)
        """
        super(VAE_Decoder, self).__init__()
        self.seq_len = seq_len
        self.features = features
        self.latent_dim = latent_dim
        self.lstm_units = lstm_units
        self.decoder = self.build_BiLSTM_decoder()

    def build_BiLSTM_decoder(self) -> tf.keras.Model:
        """Build the decoder architecture"""
        # Input layer for attention output
        attention_input = tf.keras.layers.Input(shape=(self.seq_len, self.latent_dim))
        
        # BiLSTM layers
        x = attention_input
        for units in self.lstm_units:
            x = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(units, return_sequences=True)
            )(x)
        
        # Output layers for distribution parameters
        Xhat_mean = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(self.features),
            name="Xhat_mean"
        )(x)
        
        Xhat_log_var = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(self.features),
            name="Xhat_log_var"
        )(x)
        
        # Reparameterization trick
        output_dist = tfp.distributions.Normal(loc=0., scale=1.)
        eps = output_dist.sample(tf.shape(Xhat_mean))
        Xhat = Xhat_mean + tf.sqrt(tf.math.exp(Xhat_log_var)) * eps
        
        return tf.keras.Model(
            attention_input,
            [Xhat_mean, Xhat_log_var, Xhat],
            name="decoder"
        )

    @tf.function
    def call(self, inputs: tf.Tensor, **kwargs) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Forward pass"""
        return self.decoder(inputs, **kwargs) 