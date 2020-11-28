import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

def ortho_init(scale=1.0):
    def _ortho_init(shape, dtype, partition_info=None):
        #lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4: # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    return _ortho_init

def MLP(num_state, num_actions, num_layers=2, num_hidden=64, activation=tf.tanh):
    x = tf.keras.Input(shape=num_state)
    h = x
    for i in range(num_layers):
        h = tf.keras.layers.Dense(units=num_hidden, kernel_initializer=ortho_init(np.sqrt(2)),
                                    name='mlp_fc{}'.format(i), activation=activation)(h)
    outputs = layers.Dense(num_actions, activation=activation)(x)
    model = tf.keras.Model(inputs=x, outputs=outputs)
    return model

# todo add different type of Q-func

