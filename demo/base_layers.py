from abc import ABCMeta, abstractmethod
import tensorflow as tf
from functools import reduce
import numpy as np
from typing import Union

class Layer(object, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        self.weights = []

    def get_weights(self):
        return self.weights

    @abstractmethod
    def dump(self, fp):
        pass

    @staticmethod
    def write_matrix(fp, mat):
        for x in np.nditer(mat, order='F'):
            fp.write("\t%.9g" % x)


def naive_silu(x):
    return x * tf.nn.sigmoid(x)


class Dense(Layer):
    def __init__(self, shape, act=None, norm=True, use_bias=True, name='Dense', mode='Train'):
        super().__init__()
        self.shape = shape
        self.act = act
        self.norm = norm

        self.w = tf.compat.v1.get_variable(name + '_' + mode + '_w',
                                  shape=self.shape,
                                  initializer=tf.keras.initializers.glorot_normal(),
                                  trainable=True)
        self.weights.append(self.w)
        self.b = None
        if use_bias:
            self.b = tf.compat.v1.get_variable(name + '_' + mode + '_b',
                                               shape=(1, self.shape[1]),
                                               initializer=tf.keras.initializers.zeros(),
                                               trainable=True)
            self.weights.append(self.b)

        self.lyn = None
        if norm:
            self.lyn = tf.keras.layers.LayerNormalization(axis=1, scale=False, center=False, name=name + '_' + mode)

    def __call__(self, x):
        _ret = tf.matmul(x, self.w)

        if self.b is not None:
            _ret += self.b

        if self.lyn is not None:
            _ret = self.lyn(_ret)

        if self.act is not None:
            _ret = self.act(_ret)
        return _ret

    def get_weights(self):
        return self.weights

    def dump(self, fp):
        self.write_matrix(fp, self.w)
        if self.b is not None:
            self.write_matrix(fp, self.b)

class Sequential(Layer):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
        self.weights = sum(map(lambda l: l.get_weights(), layers), [])

    def __call__(self, x):
        return reduce(lambda x, f: f(x), self.layers, x)

    def dump(self, fp):
        for layer in self.layers:
            layer.dump(fp)

class MLP(Layer):
    def __init__(self, hidden_size, act_last=False, act=tf.nn.relu, ln=False, name='MLP', mode='Train'):
        super().__init__()
        self.hidden_size = hidden_size
        self.model = Sequential([Dense((x, y), None, False, name=name+ '_' + mode+'_'+ 'Dense_' + str(i)) if not act_last and i + 1 == len(hidden_size)-1 else Dense((x, y), act, ln, name=name+ '_' + mode+'_'+'Dense' + str(i)) for i, (x, y) in enumerate(zip(hidden_size[:-1], hidden_size[1:]))])

    def __call__(self, x):
        return self.model(x)

    def get_weights(self):
        return self.model.get_weights()

    def dump(self, fp):
        self.model.dump(fp)