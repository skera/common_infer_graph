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

class Lhuc(Layer):
    def __init__(self, hidden_dim, name='Lhuc', mode='train'):
        super().__init__()
        self.h1 = tf.get_variable(name + '_' + mode + '_w1',
                                  shape=hidden_dim[:2],
                                  initializer=tf.keras.initializers.glorot_normal(),
                                  trainable=True)
        self.h2 = tf.get_variable(name + '_' + mode + '_w2',
                                  shape=hidden_dim[1:],
                                  initializer=tf.keras.initializers.glorot_normal(),
                                  trainable=True)
        self.weights += [self.h1, self.h2]

    def __call__(self, x):
        x = naive_silu(tf.matmul(x, self.h1))
        o = 2 * tf.nn.sigmoid(tf.matmul(x, self.h2))
        # output_tensor = tf.identity(o, name="output")  # 必须赋值一个变量
        return o

    def dump(self, fp):
        for w in self.weights:
            self.write_matrix(fp, w)


# 导出onnx
def export_ckpt_to_onnx(ckpt_path, pb_path, onnx_path, hidden_dim):
    tf.reset_default_graph()
    input_tensor = tf.placeholder(tf.float32, shape=[None, hidden_dim[0]], name="lhuc_input")
    lhuc_layer = Lhuc(hidden_dim, mode='train')
    output_tensor = lhuc_layer(input_tensor)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, ckpt_path)

        output_node_names = ["mul_1"]

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
        input_names=["lhuc_input:0"],
        output_names=["mul_1:0"]
    )
    model_proto = onnx_graph.make_model("Lhuc Model")
    with open(onnx_path, "wb") as f:
        f.write(model_proto.SerializeToString())
    print(f"✅ ONNX 模型已保存为:", onnx_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_dim", type=int, nargs=3, default=[320, 32, 320])
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--ckpt_path", type=str, default="../model/lhuc/lhuc.ckpt")
    parser.add_argument("--pb_path", type=str, default="../model/lhuc/lhuc.pb")
    parser.add_argument("--onnx_path", type=str, default="../model/lhuc/lhuc.onnx")
    parser.add_argument("--do_export", type=bool, default="true", help="是否导出 ONNX 模型")

    args = parser.parse_args()

    print("---------- 当前模型: Lhuc ----------")
    print("Hidden dimensions:", args.hidden_dim)
    print("Batch size:", args.batch_size)

    tf.reset_default_graph()

    input_placeholder = tf.placeholder(tf.float32, shape=[None, args.hidden_dim[0]], name="lhuc_input")
    lhuc_layer = Lhuc(args.hidden_dim, mode='train')
    output_tensor = lhuc_layer(input_placeholder)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(output_tensor, feed_dict={
            input_placeholder: np.random.normal(size=(args.batch_size, args.hidden_dim[0]))
        })
        os.makedirs(os.path.dirname(args.ckpt_path), exist_ok=True)
        save_path = saver.save(sess, args.ckpt_path)
        print("✅ 模型保存至:", save_path)

    if args.do_export:
        export_ckpt_to_onnx(args.ckpt_path, args.pb_path, args.onnx_path, args.hidden_dim)
