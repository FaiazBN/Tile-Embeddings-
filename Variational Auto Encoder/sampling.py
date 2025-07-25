from keras import layers, ops
from keras.saving import register_keras_serializable
import tensorflow as tf

@register_keras_serializable()
class Sampling(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = tf.random.Generator.from_seed(1337)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_mean)[1]
        epsilon = self.seed_generator.normal(shape=(batch, dim))
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon