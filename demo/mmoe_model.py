from abc import ABCMeta, abstractmethod
import tensorflow.compat.v1 as tf
from functools import reduce
import numpy as np
from typing import Union
import tf2onnx
import argparse
import os
from base_layers import Layer, naive_silu, MLP

tf.disable_eager_execution()
tf.disable_resource_variables()

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

    # @tf.function
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


def export_ckpt_to_onnx(ckpt_path, pb_path, onnx_path, expert_hidden_size, meta_size, expert_num):
    tf.reset_default_graph()
    input_tensor = tf.placeholder(tf.float32, shape=[None, expert_hidden_size[0]], name="mmoe_input")
    meta_tensor = tf.placeholder(tf.float32, shape=[None, meta_size], name="mmoe_meta_input")
    mmoe_layer = Tentacle_V3(expert_hidden_size, meta_size, slot_size=8, expert_num=expert_num)
    raw_output = mmoe_layer(input_tensor, meta_tensor)
    output_tensor = tf.identity(raw_output, name="mmoe_output")  # ✅ 保证输出节点存在

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, ckpt_path)

        print("\n=== Graph operations ===")
        for op in sess.graph.get_operations():
            print(op.name)

        output_node_names = ["mmoe_output"]

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
        input_names=["mmoe_input:0", "mmoe_meta_input:0"],
        output_names=["Sum:0"],
        opset=12
    )
    model_proto = onnx_graph.make_model("mmoe Model")
    with open(onnx_path, "wb") as f:
        f.write(model_proto.SerializeToString())
    print(f"✅ ONNX 模型已保存为:", onnx_path)


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
    parser.add_argument("--ckpt_path", type=str, default="../model/mmoe/mmoe.ckpt")
    parser.add_argument("--pb_path", type=str, default="../model/mmoe/mmoe.pb")
    parser.add_argument("--onnx_path", type=str, default="../model/mmoe/mmoe.onnx")
    parser.add_argument("--do_export", type=bool, default="true", help="是否导出 ONNX 模型")
    args = parser.parse_args()

    print("---------- Current model: Tentacle_V3 ----------")
    print("Expert hidden dimensions:", args.expert_hidden_size)
    print("Meta size:", args.meta_size)
    print("Batch size:", args.batch_size)
    print("Expert number:", args.expert_num)

    # # 使用命令行参数设置hidden_dim
    # expert_hidden_size = args.expert_hidden_size
    # tentacle_layer = Tentacle_V3(expert_hidden_size, args.meta_size, slot_size=8, expert_num=args.expert_num)
    # # 创建输入数据
    # x = tf.random.normal([args.batch_size, expert_hidden_size[0]])
    # meta_x = tf.random.normal([args.batch_size, args.meta_size])
    # print("Input shape:", x.shape, meta_x.shape)
    # # 计算输出
    # output = tentacle_layer(x, meta_x)
    # print("Output shape:", output.shape)

    tf.reset_default_graph()
    expert_hidden_size = args.expert_hidden_size

    tentacle_layer = Tentacle_V3(expert_hidden_size, args.meta_size, slot_size=8, expert_num=args.expert_num)

    input_placeholder = tf.placeholder(tf.float32, shape=[None, expert_hidden_size[0]], name="mmoe_input")
    meta_placeholder = tf.placeholder(tf.float32, shape=[None, args.meta_size], name="mmoe_meta_input")
    raw_output = tentacle_layer(input_placeholder, meta_placeholder)
    output_tensor = tf.identity(raw_output, name="mmoe_output")
    print("Output tensor shape:", output_tensor.shape)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(output_tensor, feed_dict={
            input_placeholder: np.random.normal(size=(args.batch_size, expert_hidden_size[0])),
            meta_placeholder: np.random.normal(size=(args.batch_size, args.meta_size))
        })
        os.makedirs(os.path.dirname(args.ckpt_path), exist_ok=True)
        save_path = saver.save(sess, args.ckpt_path)
        print("✅ 模型保存至:", save_path)

    if args.do_export:
        export_ckpt_to_onnx(args.ckpt_path, args.pb_path, args.onnx_path, expert_hidden_size, args.meta_size, args.expert_num)
