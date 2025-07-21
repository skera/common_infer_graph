from abc import ABCMeta, abstractmethod
import tensorflow.compat.v1 as tf
from functools import reduce
import numpy as np
from typing import Union
import tf2onnx
import argparse
import os
from base_layers import Layer, naive_silu, MLP, Dense
from lhuc_model import Lhuc

tf.disable_eager_execution()
tf.disable_resource_variables()

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


def export_ckpt_to_onnx(ckpt_path, pb_path, onnx_path, hidden_dim, lhuc_dim):
    tf.reset_default_graph()
    input_placeholder = tf.placeholder(tf.float32, shape=[None, hidden_dim[0]], name="ppnet_input")
    lhuc_placeholder = tf.placeholder(tf.float32, shape=[None, lhuc_dim[0]], name="lhuc_input")
    pep_block_layer = PEPBlock(hidden_dim=hidden_dim, lhuc_dim=lhuc_dim, mode='train')
    raw_output = pep_block_layer(input_placeholder, lhuc_placeholder)
    output_tensor = tf.identity(raw_output, name="ppnet_output")  # ✅ 保证输出节点存在

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, ckpt_path)

        print("\n=== Graph operations ===")
        for op in sess.graph.get_operations():
            print(op.name)

        output_node_names = ["ppnet_output"]

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
        input_names=["ppnet_input:0", "lhuc_input:0"],
        output_names=["mul_7:0"],
        opset=12
    )
    model_proto = onnx_graph.make_model("ppnet Model")
    with open(onnx_path, "wb") as f:
        f.write(model_proto.SerializeToString())
    print(f"✅ ONNX 模型已保存为:", onnx_path)


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
    parser.add_argument("--ckpt_path", type=str, default="../model/ppnet/ppnet.ckpt")
    parser.add_argument("--pb_path", type=str, default="../model/ppnet/ppnet.pb")
    parser.add_argument("--onnx_path", type=str, default="../model/ppnet/ppnet.onnx")
    parser.add_argument("--do_export", type=bool, default="true", help="是否导出 ONNX 模型")
    args = parser.parse_args()

    print("---------- Current model: PEPBlock ----------")
    print("Hidden dimensions:", args.hidden_dim)
    print("Lhuc dimensions:", args.lhuc_dim)
    print("Batch size:", args.batch_size)

    # # 使用命令行参数设置hidden_dim和lhuc_dim
    # hidden_dim = args.hidden_dim
    # lhuc_dim = args.lhuc_dim
    # pep_block_layer = PEPBlock(hidden_dim=hidden_dim, lhuc_dim=lhuc_dim, mode='train')
    
    # general_emb = tf.random.normal([args.batch_size, hidden_dim[0]])
    # lhuc_emb = tf.random.normal([args.batch_size, lhuc_dim[0]])
    
    # print("General embedding shape:", general_emb.shape)
    # print("LHUC embedding shape:", lhuc_emb.shape)
    # # 创建输入数据
    # output = pep_block_layer(general_emb, lhuc_emb)
    # print("Output shape:", output.shape)

    # tf.reset_default_graph()
    hidden_dim = args.hidden_dim
    lhuc_dim = args.lhuc_dim
    pep_block_layer = PEPBlock(hidden_dim=hidden_dim, lhuc_dim=lhuc_dim, mode='train')

    input_placeholder = tf.placeholder(tf.float32, shape=[None, hidden_dim[0]], name="ppnet_input")
    lhuc_placeholder = tf.placeholder(tf.float32, shape=[None, lhuc_dim[0]], name="lhuc_input")
    raw_output = pep_block_layer(input_placeholder, lhuc_placeholder)
    output_tensor = tf.identity(raw_output, name="ppnet_output")
    print("Output tensor shape:", output_tensor.shape)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        output = sess.run(output_tensor, feed_dict={
            input_placeholder: np.random.normal(size=(args.batch_size, hidden_dim[0])),
            lhuc_placeholder: np.random.normal(size=(args.batch_size, lhuc_dim[0]))
        })
        os.makedirs(os.path.dirname(args.ckpt_path), exist_ok=True)
        save_path = saver.save(sess, args.ckpt_path)
        print("✅ 模型保存至:", save_path)

    if args.do_export:
        export_ckpt_to_onnx(args.ckpt_path, args.pb_path, args.onnx_path, hidden_dim, lhuc_dim)
