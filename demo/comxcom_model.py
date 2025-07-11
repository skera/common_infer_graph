from abc import ABCMeta, abstractmethod
import tensorflow as tf
from functools import reduce
import numpy as np
from typing import Union
from base_layers import Layer, naive_silu
import argparse

class SENet(Layer):
    def __init__(self, hidden_size, name='SENet', mode='Train'):
        super().__init__()
        self.ln = tf.keras.layers.LayerNormalization(axis=1, scale=False, center=False)
        for i, shape in enumerate(zip(hidden_size, hidden_size[1:])):
            self.weights.append(tf.compat.v1.get_variable(name + '_' + mode + '_w_' + str(i),
                                                          shape=shape,
                                                          initializer=tf.keras.initializers.glorot_normal(),
                                                          trainable=True))
            self.weights.append(tf.compat.v1.get_variable(name + '_' + mode + '_b_' + str(i),
                                                          shape=[1, shape[1]],
                                                          initializer=tf.keras.initializers.zeros(),
                                                          trainable=True))

    def __call__(self, x):
        z = self.ln(tf.nn.relu(tf.matmul(tf.reduce_mean(x, axis=2), self.weights[0]) + self.weights[1]))
        a = tf.expand_dims(tf.nn.sigmoid(tf.matmul(z, self.weights[2]) + self.weights[3]), axis=-1)
        return a * x

    def dump(self, fp):
        for w in self.weights:
            self.write_matrix(fp, w)


class ComXCom(Layer):
    def __init__(self, cross_slot_num, other_slot_num, emb_size=16, head_num=16, name='Cross', use_se_after=False, compress_after_cross=False, use_bias=False, use_tanh=False):
        # 初始化类，定义交叉槽位数、其他槽位数、嵌入维度、头数等参数
        self.cross_slot_num = cross_slot_num  # 交叉槽位数
        self.other_slot_num = other_slot_num  # 其他槽位数
        self.total_slot_num = cross_slot_num + other_slot_num  # 总槽位数

        self.emb_size = emb_size  # 嵌入维度
        self.head_num = head_num  # 多头数量

        self.use_se_after = use_se_after  # 是否在交叉后使用SE模块
        self.compress_after_cross = compress_after_cross  # 是否在交叉后进行压缩
        self.use_bias = use_bias  # 是否使用偏置
        self.use_tanh = use_tanh  # 是否使用tanh激活函数

        # 定义压缩权重矩阵
        self.compress_w = tf.compat.v1.get_variable(name+'_compress_w',
                                                    shape=[emb_size * cross_slot_num, emb_size * head_num],
                                                    initializer=tf.keras.initializers.glorot_normal(),
                                                    trainable=True)
        # 如果使用偏置，定义偏置向量
        if self.use_bias:
            self.compress_b = tf.compat.v1.get_variable(name+'_compress_b',
                                                        shape=[emb_size * head_num],
                                                        initializer=tf.keras.initializers.glorot_normal(),
                                                        trainable=True)
        # 如果在交叉后进行压缩，定义额外的压缩权重矩阵
        if self.compress_after_cross:
            self.compress_w2 = tf.compat.v1.get_variable(name+'_compress_w2',
                                                         shape=[cross_slot_num, head_num, emb_size],
                                                         initializer=tf.keras.initializers.glorot_normal(),
                                                         trainable=True)
        # 如果使用SE模块，初始化SE网络
        if self.use_se_after:
            assert self.compress_after_cross or self.emb_size == self.head_num, "如果使用SE模块，必须设置compress_after_cross=True或emb_size == head_num！"
        self.se_dense = SENet((self.total_slot_num, self.total_slot_num // 2, self.total_slot_num), name=name+'_SENet')

    """
    x: 输入张量，形状为 [B, N, D]
    返回值: 形状为 [B, N * D|L] 的张量
    """
    def __call__(self, x):
        # 如果不在交叉后使用SE模块，则在交叉前应用SE模块
        if not self.use_se_after: 
            x = self.se_dense(x)

        # 分割输入张量并展平
        if self.other_slot_num > 0:
            C, O = tf.split(x, [self.cross_slot_num, self.other_slot_num], axis=1)  # 分割为交叉槽和其他槽
            O_flat = tf.reshape(O, [-1, self.other_slot_num * self.emb_size])  # 展平其他槽
            C_flat = tf.reshape(C, [-1, self.cross_slot_num * self.emb_size])  # 展平交叉槽, [batch, cross_slot_num * emb_size]
        else:
            C = x
            C_flat = tf.reshape(C, [-1, self.cross_slot_num * self.emb_size])   # 展平交叉槽, [batch, cross_slot_num * emb_size]

        # 压缩和交叉操作
        # C_flat: [batch, cross_slot_num * emb_size]
        # self.compress_w: [cross_slot_num * emb_size, emb_size * head_num]
        # emb_comp: [batch, emb_size * head_num]
        emb_comp = tf.matmul(C_flat, self.compress_w)  # 压缩交叉槽，形状为 [batch, dim * head_num]
        if self.use_bias: 
            emb_comp += self.compress_b  # 添加偏置
        if self.use_tanh: 
            emb_comp = tf.nn.tanh(emb_comp)  # 使用tanh激活函数

        # C: [batch, cross_slot_num, emb_size]
        # emb_comp: [batch, emb_size * head_num]
        # emb_cross: [batch, cross_slot_num, head_num]
        emb_cross = tf.matmul(C, tf.reshape(emb_comp, [-1, self.emb_size, self.head_num]))  # 交叉操作

        # 如果在交叉后进行压缩
        if self.compress_after_cross:
            # 使用einsum进行压缩，einsmum 是一种高效的张量操作方式
            # tf.einsum是一个用于执行爱因斯坦求和约定的函数，可以高效地进行张量操作
            # tf.einsum可按爱因斯坦求和约定定义张量间的乘法和求和关系，通常写成如 "bij,bjk->bik" 这种形式，左侧是输入张量的索引布局，右侧则表示输出张量的维度布局与求和方式。例如：
            # x = tf.random.normal([2, 3, 4])
            # y = tf.random.normal([2, 4, 5])
            # z = tf.einsum('bij,bjk->bik', x, y)  # 结果 shape: [2, 3, 5]
            emb_virtual = tf.einsum('bch,che->bce', emb_cross, self.compress_w2)    # [batch, cross_slot_num， emb_size]
            emb_virtual_flat = tf.reshape(emb_virtual, [-1, self.cross_slot_num * self.emb_size])  # 展平
        else:
            emb_virtual = emb_cross
            emb_virtual_flat = tf.reshape(emb_virtual, [-1, self.cross_slot_num * self.head_num])

        # 拼接交叉槽和其他槽
        if self.other_slot_num > 0:
            emb_concat = tf.concat((emb_virtual_flat, O_flat), axis=-1)
        else:
            emb_concat = emb_virtual_flat

        # 如果在交叉后使用SE模块
        if self.use_se_after:
            emb_se = self.se_dense(tf.reshape(emb_concat, [-1, self.total_slot_num, self.emb_size]))
            return tf.reshape(emb_se, [-1, self.total_slot_num * self.emb_size])
        return emb_concat

    def dump(self, fp):
        # 将权重写入文件
        pass


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Test ComXCom class")
    parser.add_argument("--cross_slot_num", type=int, default=4, help="Number of cross slots")
    parser.add_argument("--other_slot_num", type=int, default=2, help="Number of other slots")
    parser.add_argument("--emb_size", type=int, default=16, help="Embedding size")
    parser.add_argument("--head_num", type=int, default=8, help="Number of heads")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    args = parser.parse_args()

    print("---------- Current model: ComXCom ----------")
    print("Cross slot number:", args.cross_slot_num)
    print("Other slot number:", args.other_slot_num)
    print("Embedding size:", args.emb_size)
    print("Head number:", args.head_num)

    # 创建ComXCom实例并测试
    comxcom_layer = ComXCom(args.cross_slot_num, args.other_slot_num, args.emb_size, args.head_num)
    x = tf.random.normal([args.batch_size, args.cross_slot_num + args.other_slot_num, args.emb_size])
    print("Input shape:", x.shape)
    # 调用ComXCom层
    output = comxcom_layer(x)
    print("Output shape:", output.shape)