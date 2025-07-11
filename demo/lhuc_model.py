from abc import ABCMeta, abstractmethod
import tensorflow as tf
from functools import reduce
import numpy as np
from typing import Union
from base_layers import Layer, naive_silu
import argparse

# Lhuc的全称是Layer-wise Heterogeneous Unit-wise Compression，是一种用于模型压缩的技术。
# 它用来在神经网络中对每一层的权重进行压缩，以减少模型的大小和计算复杂度，同时尽量保持模型的性能。
class Lhuc(Layer):

    # hidden_dim: [H1, H2, H3]
    # input: [B, H1]
    # output: [B, H3]

    def __init__(self, hidden_dim, name='Lhuc', mode='train'):
        super().__init__()
        self.h1 = tf.compat.v1.get_variable(name + '_' + mode + '_w1',
                                  shape=hidden_dim[:2], # hidden_dim的前两个维度
                                  initializer=tf.keras.initializers.glorot_normal(),
                                  trainable=True)
        self.h2 = tf.compat.v1.get_variable(name + '_' + mode + '_w2',
                                       shape=hidden_dim[1:],    # hidden_dim的后两个维度
                                       initializer=tf.keras.initializers.glorot_normal(),
                                       trainable=True)
        self.weights += [self.h1, self.h2]

    def __call__(self, x):
        x = naive_silu(tf.matmul(x, self.h1))
        o = 2 * tf.nn.sigmoid(tf.matmul(x, self.h2))
        return o

    def dump(self, fp):
        for w in self.weights:
            self.write_matrix(fp, w)


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Test Lhuc class")
    parser.add_argument("--hidden_dim", type=int, nargs=3, 
                        help="Dimensions for the Lhuc layer, e.g., --hidden_dim 4 3 2",
                        default=[10*32, 32, 10*32])
    parser.add_argument("--batch_size", type=int, 
                        help="Batch size for input data, e.g., --batch_size 5",
                        default=1024)
    args = parser.parse_args()

    print("---------- Current model: Lhuc ----------")
    print("Hidden dimensions:", args.hidden_dim)
    print("Batch size:", args.batch_size)


    # 使用命令行参数设置hidden_dim
    hidden_dim = args.hidden_dim
    lhuc_layer = Lhuc(hidden_dim, mode='train')
    x = tf.random.normal([args.batch_size, hidden_dim[0]])
    output = lhuc_layer(x)
    print("Output shape:", output.shape)

    # 打印权重
    for weight in lhuc_layer.get_weights():
        print("Weight shape:", weight.shape)