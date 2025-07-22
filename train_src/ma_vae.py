import tensorflow as tf
import tensorflow_probability as tfp
from typing import Dict, Tuple, List
from config import *
from encoder import VAE_Encoder
from decoder import VAE_Decoder
from attention import MA

class MA_VAE(tf.keras.Model):
    def __init__(self,
                 encoder: VAE_Encoder,
                 decoder: VAE_Decoder,
                 ma: MA,
                 beta: float = INITIAL_BETA):
        """
        Initialize the MA-VAE model
        
        Args:
            encoder: Encoder model
            decoder: Decoder model
            ma: Multi-head attention model
            beta: Initial KL weight
        """
        super(MA_VAE, self).__init__()
        
        # Model components
        self.encoder = encoder
        self.decoder = decoder
        self.ma = ma
        
        # Metrics
        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.log_probs_loss_tracker = tf.keras.metrics.Mean(name="log_probs_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        
        # KL weight
        self.beta = tf.Variable(beta, trainable=False)
        
    def get_config(self):
        config = super(MA_VAE, self).get_config()
        config.update({
            'encoder': self.encoder,
            'decoder': self.decoder,
            'ma': self.ma,
            'beta': float(self.beta.numpy())
        })
        return config
        
    @classmethod
    def from_config(cls, config):
        return cls(**config)
        
    @tf.function
    def loss_fn(self,
                X: tf.Tensor,
                Xhat: tf.Tensor,
                Xhat_mean: tf.Tensor,
                Xhat_log_var: tf.Tensor,
                z_mean: tf.Tensor,
                z_log_var: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Calculate reconstruction and KL losses"""
        # Reconstruction loss (log probability)
        log_probs_loss = self.evaluate_log_prob(
            X,
            loc=Xhat_mean,
            scale=tf.sqrt(tf.math.exp(Xhat_log_var))
        )
        log_probs_loss = tf.reduce_sum(log_probs_loss, axis=1)
        
        # KL divergence
        kl_loss = tfp.distributions.kl_divergence(
            tfp.distributions.Normal(loc=0., scale=1.),
            tfp.distributions.Normal(
                loc=z_mean,
                scale=tf.sqrt(tf.math.exp(z_log_var))
            )
        )
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1), axis=-1)
        
        return -log_probs_loss, kl_loss

    @tf.function
    def evaluate_log_prob(self,
                         sample: tf.Tensor,
                         loc: tf.Tensor,
                         scale: tf.Tensor) -> tf.Tensor:
        """Evaluate log probability of sample under distribution"""
        output_dist = tfp.distributions.MultivariateNormalDiag(
            loc=loc,
            scale_diag=scale
        )
        return output_dist.unnormalized_log_prob(sample)

    @tf.function
    def train_step(self, X: tf.Tensor) -> Dict[str, tf.Tensor]:
        """Single training step"""
        if isinstance(X, tuple):
            X = X[0]
            
        with tf.GradientTape() as tape:
            # Forward pass through encoder
            z_mean, z_log_var, z, states = self.encoder(X, training=True)
            
            # Forward pass through attention
            A = self.ma([X, z], training=True)
            
            # Forward pass through decoder
            Xhat_mean, Xhat_log_var, Xhat = self.decoder(A, training=True)
            
            # Calculate losses
            log_probs_loss, kl_loss = self.loss_fn(
                X, Xhat, Xhat_mean, Xhat_log_var, z_mean, z_log_var
            )
            
            # Total loss
            total_loss = log_probs_loss + self.beta * kl_loss
            
        # Compute and apply gradients
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.log_probs_loss_tracker.update_state(log_probs_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "log_probs_loss": self.log_probs_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    @tf.function
    def test_step(self, X: tf.Tensor) -> Dict[str, tf.Tensor]:
        """Single test/validation step"""
        if isinstance(X, tuple):
            X = X[0]
            
        # Forward passes
        z_mean, z_log_var, z, states = self.encoder(X, training=False)
        A = self.ma([X, z_mean], training=False)
        Xhat_mean, Xhat_log_var, Xhat = self.decoder(A, training=False)
        
        # Calculate losses
        log_probs_loss, kl_loss = self.loss_fn(
            X, Xhat, Xhat_mean, Xhat_log_var, z_mean, z_log_var
        )
        total_loss = log_probs_loss + self.beta * kl_loss
        
        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.log_probs_loss_tracker.update_state(log_probs_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self) -> List[tf.keras.metrics.Metric]:
        """Model metrics"""
        return [
            self.total_loss_tracker,
            self.log_probs_loss_tracker,
            self.kl_loss_tracker,
        ]

    @tf.function
    def call(self, X: tf.Tensor, **kwargs) -> Tuple[tf.Tensor, ...]:
        """Model inference"""
        z_mean, z_log_var, z, states = self.encoder(X, training=False)
        A = self.ma([X, z_mean], training=False)
        Xhat_mean, Xhat_log_var, Xhat = self.decoder(A, training=False)
        return Xhat_mean, Xhat_log_var, Xhat, z_mean, z_log_var, z, A 