import os
import time
import numpy as np # linear algebra

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as K
from tensorflow.keras import initializers, regularizers, constraints, optimizers, layers, activations


embed_size = 1836 # how big is each word vector
max_features = 95000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 128 # max number of words in a question to use
n_heads = 2 # Number of heads as in Multi-head attention

class DotProdSelfAttention(Layer):
    """The self-attention layer as in 'Attention is all you need'.
    paper reference: https://arxiv.org/abs/1706.03762

    """

    def __init__(self, units,
                 activation=None,
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(DotProdSelfAttention, self).__init__(*kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) == 3
        input_dim = input_shape[-1]
        # We assume the output-dim of Q, K, V are the same
        self.kernels = dict.fromkeys(['Q', 'K', 'V'])
        for key, _ in self.kernels.items():
            self.kernels[key] = self.add_weight(shape=(input_dim, self.units),
                                                initializer=self.kernel_initializer,
                                                name='kernel_{}'.format(key),
                                                regularizer=self.kernel_regularizer,
                                                constraint=self.kernel_constraint)
        if self.use_bias:
            raise NotImplementedError
        super(DotProdSelfAttention, self).build(input_shape)

    def call(self, x):
        Q = K.dot(x, self.kernels['Q'])
        K_mat = K.dot(x, self.kernels['K'])
        V = K.dot(x, self.kernels['V'])
        attention = K.batch_dot(Q, K.permute_dimensions(K_mat, [0, 2, 1]))
        d_k = K.constant(self.units, dtype=K.floatx())
        attention = attention / K.sqrt(d_k)
        attention = K.batch_dot(K.softmax(attention, axis=-1), V)
        return attention

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)


def encoder(input_tensor):
    """One encoder as in Attention Is All You Need
    """
    # Sub-layer 1
    # Multi-Head Attention
    multiheads = []
    d_v = embed_size // n_heads
    for i in range(n_heads):
        multiheads.append(DotProdSelfAttention(d_v)(input_tensor))
    multiheads = concatenate(multiheads, axis=-1)
    multiheads = Dense(embed_size)(multiheads)
    multiheads = Dropout(0.1)(multiheads)

    # Residual Connection
    res_con = add([input_tensor, multiheads])
    # Didn't use layer normalization, use Batch Normalization instead here
    res_con = BatchNormalization(axis=-1)(res_con)

    # Sub-layer 2
    # 2 Feed forward layer
    ff1 = Dense(64, activation='relu')(res_con)
    ff2 = Dense(embed_size)(ff1)
    output = add([res_con, ff2])
    output = BatchNormalization(axis=-1)(output)

    return output

def positional_signal(hidden_size: int, length: int,
                      min_timescale: float = 1.0, max_timescale: float = 1e4):
    """
    Helper function, constructing basic positional encoding.
    The code is partially based on implementation from Tensor2Tensor library
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
    """

    if hidden_size % 2 != 0:
        raise ValueError(
            f"The hidden dimension of the model must be divisible by 2."
            f"Currently it is {hidden_size}")
    position = K.arange(0, length, dtype=K.floatx())
    num_timescales = hidden_size // 2
    log_timescale_increment = K.constant(
        (np.log(float(max_timescale) / float(min_timescale)) /
         (num_timescales - 1)),
        dtype=K.floatx())
    inv_timescales = (
            min_timescale *
            K.exp(K.arange(num_timescales, dtype=K.floatx()) *
                  -log_timescale_increment))
    scaled_time = K.expand_dims(position, 1) * K.expand_dims(inv_timescales, 0)
    signal = K.concatenate([K.sin(scaled_time), K.cos(scaled_time)], axis=1)
    return K.expand_dims(signal, axis=0)

class AddPositionalEncoding(Layer):
    """
    Injects positional encoding signal described in section 3.5 of the original
    paper "Attention is all you need". Also a base class for more complex
    coordinate encoding described in "Universal Transformers".
    """

    def __init__(self, min_timescale: float = 1.0,
                 max_timescale: float = 1.0e4, **kwargs):
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale
        self.signal = None
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config['min_timescale'] = self.min_timescale
        config['max_timescale'] = self.max_timescale
        return config

    def build(self, input_shape):
        _, length, hidden_size = input_shape
        self.signal = positional_signal(
            hidden_size, length, self.min_timescale, self.max_timescale)
        return super().build(input_shape)

    def call(self, inputs, **kwargs):
        return inputs + self.signal

def crop(dimension, start, end):
    # Crops (or slices) a Tensor on a given dimension from start to end
    # example : to crop tensor x[:, :, 5:10]
    # call slice(2, 5, 10) as you want to crop on the second dimension
    def func(x):
        if dimension == 0:
            return x[start: end]
        if dimension == 1:
            return x[:, start: end]
        if dimension == 2:
            return x[:, :, start: end]
        if dimension == 3:
            return x[:, :, :, start: end]
        if dimension == 4:
            return x[:, :, :, :, start: end]
    return Lambda(func)

def model_transformer(n_encoder=2):
    inp = Input(shape=(maxlen, embed_size))
    bert = crop(2, 0, 768*2+300)(inp)
#     bert = Masking(mask_value=0., input_shape=(maxlen, 1836))(bert)

#     # ft = crop(2, 0, 1324)(x)
#     x = crop(2, 1836, 1841)(inp)
    bert = Masking(mask_value=0., input_shape=(maxlen, embed_size))(bert)
    bert = AddPositionalEncoding()(bert)
    bert = Dropout(0.2)(bert)
    for i in range(n_encoder):
        bert = encoder(bert)
#     bert = GlobalAveragePooling1D()(bert)
#     bert = TimeDistributed(Dense(32, activation='relu'))(bert)
#     x = concatenate([bert, x])
    output = Dense(37, activation='softmax')(bert)
    model = Model(inp, output)
    
    return model
# model = model_transformer()
# model.summary()