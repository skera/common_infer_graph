from abc import ABCMeta, abstractmethod
import tensorflow as tf
from functools import reduce
import numpy as np
from typing import Union
from base_layers import Layer, naive_silu, Dense
from lhuc_model import Lhuc
import argparse


class PEPBlock(Layer):
    def __init__(self, hidden_dim, lhuc_dim, act_last=True, name='PEPBlock', mode='train'):
        super().__init__()
        self.lhuc_layers = []
        self.mlp_layers = []
        for i, [x, y] in enumerate(zip(hidden_dim[:-1], hidden_dim[1:])):
            self.lhuc_layers.append(Lhuc(hidden_dim=[hidden_dim[0] + lhuc_dim[0], lhuc_dim[1], x], name=name + '_Lhuc_' + str(i)))
            if i == len(hidden_dim) - 2 and not act_last:
                self.mlp_layers.append(Dense((x, y), act=None, norm=False, name=name + '_' + mode + '_' + 'Dense_' + str(i)))
            else:
                self.mlp_layers.append(Dense((x, y), act=naive_silu, norm=True, name=name+ '_' + mode+'_'+ 'Dense_' + str(i)))
        self.weights += sum(map(lambda x: x.get_weights(), self.lhuc_layers + self.mlp_layers), [])

    def __call__(self, general_emb, lhuc_emb):
        lhuc_input = tf.concat([tf.stop_gradient(general_emb), lhuc_emb], axis=-1)
        hidden = general_emb
        for i in range(len(self.mlp_layers)):
            h_w = self.lhuc_layers[i](lhuc_input)
            hidden = self.mlp_layers[i](hidden * h_w)

        return hidden

    def dump(self, fp):
        for w in self.weights:
            self.write_matrix(fp, w)


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Test PEPBlock class")
    parser.add_argument("--hidden_dim", type=int, nargs=3, 
                        help="Dimensions for the PEPBlock layer, e.g., --hidden_dim 4 3 2",
                        default=[10*32, 32, 10*32])
    parser.add_argument("--lhuc_dim", type=int, nargs=2, 
                        help="Dimensions for the Lhuc layer, e.g., --lhuc_dim 4 3",
                        default=[10*32, 32])
    parser.add_argument("--batch_size", type=int, 
                        help="Batch size for input data, e.g., --batch_size 5",
                        default=1024)
    args = parser.parse_args()

    print("---------- Current model: PEPBlock ----------")
    print("Hidden dimensions:", args.hidden_dim)
    print("Lhuc dimensions:", args.lhuc_dim)
    print("Batch size:", args.batch_size)

    # 使用命令行参数设置hidden_dim和lhuc_dim
    hidden_dim = args.hidden_dim
    lhuc_dim = args.lhuc_dim
    pep_block_layer = PEPBlock(hidden_dim=hidden_dim, lhuc_dim=lhuc_dim, mode='train')
    
    general_emb = tf.random.normal([args.batch_size, hidden_dim[0]])
    lhuc_emb = tf.random.normal([args.batch_size, lhuc_dim[0]])
    
    print("General embedding shape:", general_emb.shape)
    print("LHUC embedding shape:", lhuc_emb.shape)
    # 创建输入数据
    output = pep_block_layer(general_emb, lhuc_emb)
    print("Output shape:", output.shape)