import tensorflow as tf
from typing import Tuple
from config import *

class MA(tf.keras.Model):
    def __init__(self,
                 seq_len: int = WINDOW_SIZE,
                 latent_dim: int = LATENT_DIM,
                 features: int = REDUCED_FEATURES,
                 num_heads: int = ATTENTION_HEADS,
                 key_dim: int = ATTENTION_KEY_DIM):
        """
        Initialize the Multi-head Attention module
        
        Args:
            seq_len: Length of input sequence
            latent_dim: Dimension of latent space
            features: Number of input features
            num_heads: Number of attention heads
            key_dim: Dimension of key in attention
        """
        super(MA, self).__init__()
        self.seq_len = seq_len
        self.features = features
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.ma = self.build_MA()

    def build_MA(self) -> tf.keras.Model:
        """Build the multi-head attention architecture"""
        # Create attention layer
        attention = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            output_shape=self.latent_dim,
            name="A_det"
        )

        # Input layers
        ma_input = tf.keras.layers.Input(shape=(self.seq_len, self.features))
        latent_input = tf.keras.layers.Input(shape=(self.seq_len, self.latent_dim))
        
        # Apply attention
        # Query and Key from input, Value from latent
        A = attention(
            query=ma_input,
            key=ma_input,
            value=latent_input
        )
        
        return tf.keras.Model(
            [ma_input, latent_input],
            A,
            name="MA"
        )

    @tf.function
    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor], **kwargs) -> tf.Tensor:
        """Forward pass"""
        return self.ma(inputs, **kwargs) 