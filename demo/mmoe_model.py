from abc import ABCMeta, abstractmethod
import tensorflow as tf
from functools import reduce
import numpy as np
from typing import Union
from base_layers import Layer, MLP, naive_silu
import argparse


class Gate_v2(Layer):
    def __init__(self, shape, name='Gate', mode='Train'):
        super().__init__()
        self.w1 = tf.compat.v1.get_variable(name + '_' + mode + '_w1',
                                  shape=shape,
                                  initializer=tf.keras.initializers.glorot_normal(),
                                  trainable=True)
        self.b1 = tf.compat.v1.get_variable(name + '_' + mode + '_b1',
                                  shape=(1, shape[1]),
                                  initializer=tf.keras.initializers.zeros(),
                                  trainable=True)
        self.weights.append(self.w1)
        self.weights.append(self.b1)
    def __call__(self, x, drop_rate = 0):
        if drop_rate > 0:
            return tf.nn.softmax(tf.nn.dropout(tf.matmul(x, self.w1) + self.b1, rate = drop_rate))
        return tf.nn.softmax(tf.matmul(x, self.w1) + self.b1)

    def dump(self, fp):
        for w in self.weights:
            self.write_matrix(fp, w)

# Gated Linear Unit (GLU) for slot-wise attention
class GluSlotwise(Layer):
    def __init__(self, hidden_dim, name='GluSlotwise'):
        super().__init__()
        self.h1 = tf.compat.v1.get_variable(name + '_w1',
                                  shape=hidden_dim,
                                  initializer=tf.keras.initializers.glorot_normal(),
                                  trainable=True)
        self.weights += [self.h1]

    def __call__(self, x, slot_num):
        x_w = tf.sigmoid(tf.matmul(x, self.h1)) * 2
        # [B, S] -> [B, S, 1]
        x_w = tf.expand_dims(x_w, axis=-1)
        # [B, S*E] -> [B, S, E]
        slot_size = x.shape[1] // slot_num
        x_r = tf.reshape(x, [-1, slot_num, slot_size])
        o = x_r * x_w
        # [B, S, E] -> [B, S*E]
        o_r = tf.reshape(o, [-1, slot_num * slot_size])
        return o_r

    def dump(self, fp):
        pass

class Tentacle_V3(Layer):
    def __init__(self, expert_hidden_size, meta_size, slot_size = 8, expert_num = 4, name='tentacle_v2'):
        super().__init__()
        self.experts = []
        self.lhucs = []
        self.expert_num = expert_num    # N_expert_num = 4
        expert_input_size = expert_hidden_size[0]
        self.slot_num = expert_input_size // slot_size
        self.gate = Gate_v2(shape=[meta_size, expert_num], name=name + '_gate')
        for i in range(expert_num):
            self.experts.append(MLP(expert_hidden_size, act_last=True, act=tf.nn.relu, ln=False, name = name + '_expert_' + str(i)))
            self.lhucs.append(GluSlotwise(hidden_dim=[expert_input_size, self.slot_num], name = name + '_lhuc_' + str(i)))

    @tf.function
    def __call__(self, x, meta_x, open_drop = False, drop_rate = 0.1):

        # input x: [B, S*E]
        # input meta_x: [B, M]

        lhuc_outs = [self.lhucs[i](x, self.slot_num) for i in range(self.expert_num)]       # [N_expert_num, B, S, E]
        expert_outs = tf.stack([self.experts[i](lhuc_outs[i]) for i in range(self.expert_num)], axis=1) # [B, N_expert_num, out_dim]
        gate_drop_rate = drop_rate if open_drop else 0  # drop_rate = 0.1
        gate_out = self.gate(meta_x, gate_drop_rate)    # [B, N_expert_num]
        gate_out = tf.expand_dims(gate_out, axis=-1)    # [B, N_expert_num, 1]
        gated_outs = expert_outs * gate_out # [B, N_expert_num, out_dim]
        return tf.reduce_sum(gated_outs, axis=1)    # [B, out_dim]

    def dump(self, fp):
        pass


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Test Tentacle_V3 class")
    parser.add_argument("--expert_hidden_size", type=int, nargs=3, 
                        help="Dimensions for the expert hidden layer, e.g., --expert_hidden_size 32 64 128",
                        default=[32*8, 64, 128])
    parser.add_argument("--meta_size", type=int, 
                        help="Size of the meta input, e.g., --meta_size 16",
                        default=16)
    parser.add_argument("--batch_size", type=int, 
                        help="Batch size for input data, e.g., --batch_size 5",
                        default=1024)
    parser.add_argument("--expert_num", type=int, 
                        help="Number of experts, e.g., --expert_num 4",
                        default=8)
    args = parser.parse_args()

    print("---------- Current model: Tentacle_V3 ----------")
    print("Expert hidden dimensions:", args.expert_hidden_size)
    print("Meta size:", args.meta_size)
    print("Batch size:", args.batch_size)
    print("Expert number:", args.expert_num)

    # 使用命令行参数设置hidden_dim
    expert_hidden_size = args.expert_hidden_size
    tentacle_layer = Tentacle_V3(expert_hidden_size, args.meta_size, slot_size=8, expert_num=args.expert_num)
    # 创建输入数据
    x = tf.random.normal([args.batch_size, expert_hidden_size[0]])
    meta_x = tf.random.normal([args.batch_size, args.meta_size])
    print("Input shape:", x.shape, meta_x.shape)
    # 计算输出
    output = tentacle_layer(x, meta_x)
    print("Output shape:", output.shape)
