from abc import ABCMeta, abstractmethod
import tensorflow.compat.v1 as tf
from functools import reduce
import numpy as np
from typing import Union
import tf2onnx
import argparse
import os
from base_layers import Layer, naive_silu

tf.disable_eager_execution()
tf.disable_resource_variables()


# SENet的全称是Squeeze-and-Excitation Network，是一种用于神经网络的注意力机制。
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

# MLCC的全称是Multi-Level Compression Cross，是一种用于模型压缩的技术。
# 它通过多层次的压缩和交叉操作来减少模型的大小和计算复杂度，同时尽量保持模型的性能。
class MLCC_V2(Layer):
    def __init__(
            self,
            cross_slot_num, # 传入S
            emb_size=16,    # 传入32
            head_num=64,    # 传入4
            dmlp_act_func='',
            dmlp_dim_list=[1],  # 传入[4,1]
            dmlp_dim_concat=[], # 传入[0,1]
            emb_size2=None, # 传入8
            name='MLCC_V2'
    ):
        self.cross_slot_num = cross_slot_num
        self.emb_size = emb_size    # emb_size = 32
        self.head_num = head_num    # head_num = 4
        # dyanmic mlp
        self.dmlp_param_dict = self.calc_dmlp_param(self.emb_size, dmlp_dim_list)   # {'dim_list': [32, 4, 1], 'cnt_list': [128, 4]}
        self.dmlp_cnt_sum = sum(self.dmlp_param_dict['cnt_list'])   # 128 + 4 = 132
        self.dmlp_dim_last = self.dmlp_param_dict['dim_list'][-1]   # 1
        self.dmlp_act_func = dmlp_act_func
        self.dmlp_dim_concat = dmlp_dim_concat  # [0,1]
        # dmlp_dim_concat: 0表示concat一次，1表示concat by head
        self.dmlp_concat_dim_bottom = 0  # concat once
        self.dmlp_concat_dim_top = 0     # concat by head
        if self.dmlp_dim_concat:
            assert len(dmlp_dim_concat) < len(self.dmlp_param_dict['dim_list'])
            # e.g. dim_list=[8,4,1]
            # e.g. dmlp_dim_concat=[0] -> [[0,8]]
            # e.g. dmlp_dim_concat=[1] -> [[1,4]]
            # e.g. dmlp_dim_concat=[0,1] -> [[0,8], [1,4]]
            for _dim in dmlp_dim_concat:
                concat_dim = self.dmlp_param_dict['dim_list'][_dim] # 
                if _dim == 0:
                    self.dmlp_concat_dim_bottom = concat_dim # dmlp_concat_dim_bottom = dmlp_param_dict['dim_list'][0] = 32
                else:
                    self.dmlp_concat_dim_top += concat_dim  # dmlp_concat_dim_top = dmlp_param_dict['dim_list'][1] = 4
        self.dmlp_dim_ext = self.dmlp_dim_last + self.dmlp_concat_dim_top   # dmlp_dim_ext = 1 + 4 = 5
        self.dmlp_out_dim = self.head_num * self.dmlp_dim_ext + self.dmlp_concat_dim_bottom # dmlp_out_dim = 4 * 5 + 32 = 20 + 32 = 52
        self.emb_size2 = emb_size2 if emb_size2 else self.emb_size  # emb_size2 = 8
        # [S*E, H*M]
        self.compress_l1 = tf.compat.v1.get_variable(
            name+'_compress_l1',
            shape=[self.emb_size * self.cross_slot_num, self.head_num * self.dmlp_cnt_sum], # [32*S, 4*132] = [32*S, 528]
            initializer=tf.keras.initializers.glorot_normal(),
            trainable=True
        )
        # [S, H*(Mo+Ct)+Cb, E]
        self.compress_l2 = tf.compat.v1.get_variable(
            name + '_compress_l2',
            shape=[self.cross_slot_num, self.dmlp_out_dim, self.emb_size2], # [S, 52, 8]
            initializer=tf.keras.initializers.glorot_normal(),
            trainable=True
        )
        self.se_dense = SENet((self.cross_slot_num, self.cross_slot_num // 2, self.cross_slot_num), name=name+'_SENet')

    # 计算dmlp参数, dim_list表示每一层的输出维度，cnt_list表示每一层的参数个数
    def calc_dmlp_param(self, input_size, dim_list):

        # input_size = emb_size = 32
        # dim_list = [4, 1]

        param_dim_list = [input_size] + dim_list    # [32, 4, 1]
        param_cnt_list = [x*y for x,y in zip(param_dim_list, param_dim_list[1:])]   # [32*4, 4*1] = [128, 4]
        param_dict = {
            'dim_list': param_dim_list,
            'cnt_list': param_cnt_list,
        }
        return param_dict   # {'dim_list': [32, 4, 1], 'cnt_list': [128, 4]}

    def __call__(self, x):
        # preprocess
        emb_raw_flat = x    # [B, S*E]
        emb_other_flat = None
        emb_raw = tf.reshape(x, [-1, self.cross_slot_num, self.emb_size])  # dim3   [B, S, E]

        # compress_l1
        # [B, S*E], [S*E, H*M] -> [B, H*M]
        emb_comp = tf.matmul(emb_raw_flat, self.compress_l1)    # [B, H*M]
        # [B, H*M] -> [B, H, M]
        emb_comp_r = tf.reshape(emb_comp, [-1, self.head_num, self.dmlp_cnt_sum])   # [B, H, M]

        # cross
        # [B, H, M] -> [B, H, M1], [B, H, M2], ..., [B, H, Mo]
        # self.dmlp_param_dict['cnt_list'] : # [128, 4]
        dmlp_weight_list = tf.split(emb_comp_r, self.dmlp_param_dict['cnt_list'], axis=-1)  # [B, H, 128], [B, H, 4]
        hidden = emb_raw  # [B, S, E]
        hidden_concat_list = []
        for idx, _dmlp_weight in enumerate(dmlp_weight_list):
            # avoid idx=0 (concat once)
            if idx > 0 and idx in self.dmlp_dim_concat:
                hidden_concat_list.append(hidden)
            i_dim = self.dmlp_param_dict['dim_list'][idx]
            o_dim = self.dmlp_param_dict['dim_list'][idx+1]
            # [B, H, Mx] -> [B, H, I, O]
            dmlp_weight = tf.reshape(_dmlp_weight, [-1, self.head_num, i_dim, o_dim])
            if idx == 0:
                # [B, S, I], [B, H, I, O] -> [B, H, S, O]
                hidden = tf.einsum('bsi,bhio->bhso', hidden, dmlp_weight)
            else:
                # [B, H, S, I], [B, H, I, O] -> [B, H, S, O]
                # hidden = tf.einsum('bhsi,bhio->bhso', hidden, dmlp_weight)
                hidden = tf.matmul(hidden, dmlp_weight)
            # no activation in last layer
            if idx != len(dmlp_weight_list) - 1:    # len(dmlp_weight_list) - 1 = 1
                if self.dmlp_act_func == 'relu':
                    hidden = tf.nn.relu(hidden)
                elif self.dmlp_act_func == 'tanh':
                    hidden = tf.nn.tanh(hidden)
                elif self.dmlp_act_func == 'sigmoid':
                    hidden = tf.nn.sigmoid(hidden)
        # concat top
        if len(hidden_concat_list) > 0:
            hidden_concat_list.append(hidden)
            emb_cross = tf.concat(hidden_concat_list, axis=-1)  # [B, H, S, Mo+Ct]
        else:
            emb_cross = hidden  # [B, H, S, Mo]
        # [B, H, S, Mo+Ct] -> [B, S, H, Mo+Ct] -> [B, S, H*(Mo+Ct)]
        emb_cross_t = tf.transpose(emb_cross, perm=[0, 2, 1, 3])
        emb_cross_r = tf.reshape(emb_cross_t, [-1, self.cross_slot_num, self.head_num * self.dmlp_dim_ext])
        # concat bottom
        if 0 in self.dmlp_dim_concat:
            # [B, S, H*(Mo+Ct)], [B, S, E] -> [B, S, H*(Mo+Ct)+Cb] (Cb=E)
            emb_cross_r = tf.concat([emb_cross_r, emb_raw], axis=-1)

        # compress_l2
        # [B, S, H*(Mo+Ct)+Cb], [S, H*(Mo+Ct)+Cb, E] -> [B, S, E]
        emb_refined = tf.einsum('bch,che->bce', emb_cross_r, self.compress_l2)
        # [B, S, E] -> [B, S*E]
        emb_refined_flat = tf.reshape(emb_refined, [-1, self.cross_slot_num * self.emb_size2])

        # se
        emb_concat = emb_refined_flat
        emb_se = self.se_dense(tf.reshape(emb_concat, [-1, self.cross_slot_num, self.emb_size2]))
        res = tf.reshape(emb_se, [-1, self.cross_slot_num * self.emb_size2])

        return res

    def dump(self, fp):
        pass


def export_ckpt_to_onnx(ckpt_path, pb_path, onnx_path, input_size):
    tf.reset_default_graph()
    input_tensor = tf.placeholder(tf.float32, shape=[None, input_size[1]], name="mlcc_input")
    mlcc_layer = MLCC_V2(
        cross_slot_num=args.cross_slot_num,
        emb_size=args.emb_size,
        head_num=args.head_num,
        dmlp_act_func=args.dmlp_act_func,
        dmlp_dim_list=args.dmlp_dim_list,
        dmlp_dim_concat=args.dmlp_dim_concat,
        emb_size2=args.emb_size2
    )
    raw_output = mlcc_layer(input_tensor)
    output_tensor = tf.identity(raw_output, name="mlcc_output")  # ✅ 保证输出节点存在

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, ckpt_path)

        print("\n=== Graph operations ===")
        for op in sess.graph.get_operations():
            print(op.name)

        output_node_names = ["mlcc_output"]

        frozen_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, output_node_names)
        freeze_graph_def = tf.graph_util.remove_training_nodes(frozen_graph_def)

        print("\n=== Graph outputs after freeze ===")
        for op in freeze_graph_def.node:
            if "output" in op.name:
                print(op.name)

        with tf.gfile.GFile(pb_path, "wb") as f:
            f.write(freeze_graph_def.SerializeToString())
        print("✅ 成功生成冻结变量的推理图，已保存为:", pb_path)

    with tf.gfile.GFile(pb_path, "rb") as f:
        frozen_graph_def = tf.GraphDef()
        frozen_graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(frozen_graph_def, name="")

    # 确认 node 名字是否存在
    for op in graph.get_operations():
        print(op.name)

    onnx_graph = tf2onnx.tfonnx.process_tf_graph(
        graph,
        input_names=["mlcc_input:0"],
        output_names=["Reshape_7:0"]
    )
    model_proto = onnx_graph.make_model("mlcc Model")
    with open(onnx_path, "wb") as f:
        f.write(model_proto.SerializeToString())
    print(f"✅ ONNX 模型已保存为:", onnx_path)

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Test MLCC_V2 class")
    parser.add_argument("--cross_slot_num", type=int, default=10, help="Number of cross slots, e.g., --cross_slot_num 10")
    parser.add_argument("--emb_size", type=int, default=32, help="Embedding size, e.g., --emb_size 32")
    parser.add_argument("--head_num", type=int, default=4, help="Number of heads, e.g., --head_num 4")
    parser.add_argument("--dmlp_act_func", type=str, default='relu', help="Activation function for DMLP, e.g., --dmlp_act_func relu")
    parser.add_argument("--dmlp_dim_list", type=int, nargs='+', default=[4, 1], help="Dimensions for DMLP layers, e.g., --dmlp_dim_list 4 1")
    parser.add_argument("--dmlp_dim_concat", type=int, nargs='+', default=[0], help="Dimensions to concatenate in DMLP layers, e.g., --dmlp_dim_concat 0")
    parser.add_argument("--emb_size2", type=int, default=8, help="Second embedding size, e.g., --emb_size2 8")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for input data, e.g., --batch_size 32")
    parser.add_argument("--ckpt_path", type=str, default="../model/mlcc/mlcc.ckpt")
    parser.add_argument("--pb_path", type=str, default="../model/mlcc/mlcc.pb")
    parser.add_argument("--onnx_path", type=str, default="../model/mlcc/mlcc.onnx")
    parser.add_argument("--do_export", type=bool, default="true", help="是否导出 ONNX 模型")
    args = parser.parse_args()

    print("---------- Current model: MLCC_V2 ----------")
    print("Cross slot number:", args.cross_slot_num)
    print("Embedding size:", args.emb_size)
    print("Head number:", args.head_num)
    print("DMLP activation function:", args.dmlp_act_func)
    print("DMLP dimension list:", args.dmlp_dim_list)
    print("DMLP dimension concat:", args.dmlp_dim_concat)
    print("Second embedding size:", args.emb_size2)
    
    # x = tf.random.normal([args.batch_size, args.cross_slot_num * args.emb_size])
    # print("Input shape:", x.shape)
    # # 调用MLCC_V2层
    # output = mlcc_layer(x)
    # print("Output shape:", output.shape)

    tf.reset_default_graph()

    mlcc_layer = MLCC_V2(
        cross_slot_num=args.cross_slot_num,
        emb_size=args.emb_size,
        head_num=args.head_num,
        dmlp_act_func=args.dmlp_act_func,
        dmlp_dim_list=args.dmlp_dim_list,
        dmlp_dim_concat=args.dmlp_dim_concat,
        emb_size2=args.emb_size2
    )

    input_placeholder = tf.placeholder(tf.float32, shape=[None, args.cross_slot_num * args.emb_size], name="mlcc_input")
    raw_output = mlcc_layer(input_placeholder)
    output_tensor = tf.identity(raw_output, name="mlcc_output")
    print("Output tensor shape:", output_tensor.shape)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(output_tensor, feed_dict={
            input_placeholder: np.random.normal(size=(args.batch_size, args.cross_slot_num * args.emb_size))
        })
        os.makedirs(os.path.dirname(args.ckpt_path), exist_ok=True)
        save_path = saver.save(sess, args.ckpt_path)
        print("✅ 模型保存至:", save_path)

    if args.do_export:
        export_ckpt_to_onnx(args.ckpt_path, args.pb_path, args.onnx_path, [None, args.cross_slot_num * args.emb_size])