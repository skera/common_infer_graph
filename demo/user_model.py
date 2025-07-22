import os
import tensorflow.compat.v1 as tf
import numpy as np
import argparse
from base_layers import Layer, naive_silu, MLP
from lhuc_model import Lhuc
from mlcc_model import MLCC_V2
from mmoe_model import Tentacle_V3
import tf2onnx

tf.disable_eager_execution()
tf.disable_resource_variables()

M_DICT = {
    'emb_size': 32,
    'emb_size_double': 64,
    'head_num': 4,
    'dmlp_dim_list': [4, 1],
    'dmlp_dim_concat': [0, 1],
    'emb_size2': 8
}
DELAY_BUCKETS = 19

def build_monotonic_delay_logits(original_logits):
    initial = original_logits[:, :1]
    deltas = tf.math.softplus(original_logits[:, 1:])
    cum_logits = tf.cumsum(tf.concat([initial, deltas], axis=1), axis=1)
    return cum_logits

class UserModel(Layer):
    def __init__(self, batch_size, emb_slot_list):
        super().__init__()
        self.batch_size = batch_size
        (
            self.cross_slots,
            self.other_slots,
            self.pp_add_slots,
            self.st_add_slots,
            self.sc_add_slots,
            self.lhuc_scene_slots,
            self.tentacle_gate_slots
        ) = emb_slot_list

        self.cross_emb_ph = tf.placeholder(tf.float32, [None, self.cross_slots * M_DICT['emb_size_double']], name="cross_emb")
        self.other_emb_ph = tf.placeholder(tf.float32, [None, self.other_slots * M_DICT['emb_size_double']], name="other_emb")
        self.pp_add_emb_ph = tf.placeholder(tf.float32, [None, self.pp_add_slots * M_DICT['emb_size_double']], name="pp_add_emb")
        self.st_add_emb_ph = tf.placeholder(tf.float32, [None, self.st_add_slots * M_DICT['emb_size_double']], name="st_add_emb")
        self.sc_add_emb_ph = tf.placeholder(tf.float32, [None, self.sc_add_slots * M_DICT['emb_size_double']], name="sc_add_emb")
        self.lhuc_scene_emb_ph = tf.placeholder(tf.float32, [None, self.lhuc_scene_slots * M_DICT['emb_size_double']], name="lhuc_scene_emb")
        self.tentacle_gate_emb_ph = tf.placeholder(tf.float32, [None, self.tentacle_gate_slots * M_DICT['emb_size_double']], name="tentacle_gate_emb")
        self.dsif_gate_emb_ph = tf.placeholder(tf.float32, [None, M_DICT['emb_size']], name="dsif_gate_emb")

    def construct_gpu_model(self, is_train=False):
        def emb16_split(emb16):
            slots = tf.shape(emb16)[1] // M_DICT['emb_size_double']
            emb16_t = tf.reshape(emb16, (-1, slots, M_DICT['emb_size_double']))
            emb1, emb2 = tf.split(emb16_t, [M_DICT['emb_size'], M_DICT['emb_size']], axis=2)
            return tf.reshape(emb1, (-1, slots * M_DICT['emb_size'])), tf.reshape(emb2, (-1, slots * M_DICT['emb_size']))

        cross_emb, cross_s_emb = emb16_split(self.cross_emb_ph)
        other_emb, other_s_emb = emb16_split(self.other_emb_ph)
        pp_add_emb, pp_add_s_emb = emb16_split(self.pp_add_emb_ph)
        st_add_emb, st_add_s_emb = emb16_split(self.st_add_emb_ph)
        sc_add_emb, sc_add_s_emb = emb16_split(self.sc_add_emb_ph)
        lhuc_scene_emb, lhuc_scene_s_emb = emb16_split(self.lhuc_scene_emb_ph)
        tentacle_gate_emb, tentacle_gate_s_emb = emb16_split(self.tentacle_gate_emb_ph)
        dsif_gate_emb = self.dsif_gate_emb_ph

        merge_emb = tf.concat([cross_emb, other_emb], axis=-1)
        merge_s_emb = tf.concat([cross_s_emb, other_s_emb], axis=-1)
        scene_add_emb = tf.concat([pp_add_emb, st_add_emb, sc_add_emb], axis=-1)
        scene_add_s_emb = tf.concat([pp_add_s_emb, st_add_s_emb, sc_add_s_emb], axis=-1)

        shared_slots_num = self.cross_slots + self.other_slots
        cross_deep = MLCC_V2(shared_slots_num, emb_size=M_DICT['emb_size'], head_num=M_DICT['head_num'],
                             dmlp_dim_list=M_DICT['dmlp_dim_list'], dmlp_dim_concat=M_DICT['dmlp_dim_concat'],
                             emb_size2=M_DICT['emb_size2'], name='Cross_deep')
        cross_shallow = MLCC_V2(shared_slots_num, emb_size=M_DICT['emb_size'], head_num=M_DICT['head_num'],
                                dmlp_dim_list=M_DICT['dmlp_dim_list'], dmlp_dim_concat=M_DICT['dmlp_dim_concat'],
                                emb_size2=M_DICT['emb_size2'], name='Cross_shallow')

        g_hidden = {
            'normal_bottom': [512, 256, 128, 64],
            'tentacle_bottom': [512, 256, 128, 64],
            'top': [128, 64, 1],
            'delay': [64, 32, DELAY_BUCKETS],
            'lhuc': 32,
            'scene_add': [64]
        }

        lhuc_dsif = Lhuc([g_hidden['tentacle_bottom'][-1] + g_hidden['normal_bottom'][-1] + M_DICT['emb_size'],
                          g_hidden['lhuc'], g_hidden['tentacle_bottom'][-1] + g_hidden['normal_bottom'][-1]], name='Lhuc_dsif')
        lhuc_scene = Lhuc([lhuc_scene_emb.shape[1], g_hidden['lhuc'], merge_emb.shape[1]], name='Lhuc_scene')
        lhuc_scene_shallow = Lhuc([lhuc_scene_s_emb.shape[1], g_hidden['lhuc'], merge_s_emb.shape[1]], name='Lhuc_scene_shallow')

        scene_proj_d = MLP([scene_add_emb.shape[1]] + g_hidden['scene_add'], act=tf.nn.relu, act_last=True, name='scene_fea_proj_deep')
        scene_proj_s = MLP([scene_add_s_emb.shape[1]] + g_hidden['scene_add'], act=tf.nn.relu, act_last=True, name='scene_fea_proj_shallow')
        top_dense_d = MLP([g_hidden['tentacle_bottom'][-1] + g_hidden['normal_bottom'][-1]] + g_hidden['top'], act=tf.nn.relu, act_last=False, name='deep_top')
        top_dense_s = MLP([g_hidden['normal_bottom'][-1]] + g_hidden['top'], act=tf.nn.relu, act_last=False, name='shallow_top')

        shallow_lhuc = lhuc_scene_shallow(lhuc_scene_s_emb)
        shared_s = cross_shallow(merge_s_emb * shallow_lhuc)
        bottom_input_s = tf.concat([shared_s, scene_proj_s(scene_add_s_emb)], axis=-1)
        tentacle_s = Tentacle_V3([bottom_input_s.shape[1]] + g_hidden['tentacle_bottom'], tentacle_gate_s_emb.shape[1], slot_size=8, expert_num=4, name='tentacle_s')
        bottom_out_s = tentacle_s(bottom_input_s, tentacle_gate_s_emb)
        shallow_logits = top_dense_s(bottom_out_s)
        shallow_score = tf.sigmoid(shallow_logits)

        deep_lhuc = lhuc_scene(lhuc_scene_emb)
        shared_d = cross_deep(merge_emb * deep_lhuc)
        bottom_input_d = tf.concat([shared_d, scene_proj_d(scene_add_emb)], axis=-1)
        tentacle_d = Tentacle_V3([bottom_input_d.shape[1]] + g_hidden['tentacle_bottom'], tentacle_gate_emb.shape[1], slot_size=8, expert_num=4, name='tentacle_d')
        bottom_out_d = tentacle_d(bottom_input_d, tentacle_gate_emb)

        ds_lhuc_out = lhuc_dsif(tf.concat([tf.stop_gradient(bottom_out_d), tf.stop_gradient(bottom_out_s), dsif_gate_emb], axis=-1))
        fusion_input = tf.concat([bottom_out_d, tf.stop_gradient(bottom_out_s)], axis=-1) * ds_lhuc_out
        deep_logits = top_dense_d(fusion_input)

        def back_sample(x, rate):
            return 1 / ((1 / tf.clip_by_value(x, 1e-7, 1.0 - 1e-7) - 1) / rate + 1)

        fd_out = tf.sigmoid(deep_logits)
        st_out = back_sample(tf.sigmoid(deep_logits), 0.1)

        return {
            'fd_cvr': tf.identity(fd_out, name='FD_SCORE'),
            'st_cvr': tf.identity(st_out, name='ST_SCORE')
        }

    def __call__(self):
        with tf.variable_scope("UserModel"):
            return self.construct_gpu_model(is_train=False)
        
    def dump(self, fp):
        for w in self.weights:
            self.write_matrix(fp, w)


def export_ckpt_to_onnx(ckpt_path, pb_path, onnx_path):
    tf.reset_default_graph()

    # 重新构造模型图
    model = UserModel(batch_size=1024, emb_slot_list=[10]*7)
    output_scores = model()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, ckpt_path)

        # print("\n=== Graph operations ===")
        # for op in sess.graph.get_operations():
        #     print(op.name)

        output_node_names = ["UserModel/FD_SCORE", "UserModel/ST_SCORE"]

        frozen_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, output_node_names)
        freeze_graph_def = tf.graph_util.remove_training_nodes(frozen_graph_def)

        print("\n=== Graph outputs after freeze ===")
        for op in freeze_graph_def.node:
            if "output" in op.name or "SCORE" in op.name:
                print(op.name)

        with tf.gfile.GFile(pb_path, "wb") as f:
            f.write(freeze_graph_def.SerializeToString())
        print("✅ 成功生成冻结变量的推理图，已保存为:", pb_path)

    # ONNX 转换
    with tf.gfile.GFile(pb_path, "rb") as f:
        frozen_graph_def = tf.GraphDef()
        frozen_graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(frozen_graph_def, name="")

    onnx_graph = tf2onnx.tfonnx.process_tf_graph(
        graph,
        input_names=["cross_emb:0", "other_emb:0", "pp_add_emb:0", "st_add_emb:0",
                     "sc_add_emb:0", "lhuc_scene_emb:0", "tentacle_gate_emb:0", "dsif_gate_emb:0"],
        output_names=["UserModel/Sigmoid_17:0", "UserModel/truediv_2:0"],
        opset=12
    )
    model_proto = onnx_graph.make_model("user Model")
    with open(onnx_path, "wb") as f:
        f.write(model_proto.SerializeToString())
    print(f"✅ ONNX 模型已保存为:", onnx_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--emb_slots", type=int, nargs=7, default=[10]*7)
    parser.add_argument("--ckpt_path", type=str, default="../model/user_model/user_model.ckpt")
    parser.add_argument("--pb_path", type=str, default="../model/user_model/user_model.pb")
    parser.add_argument("--onnx_path", type=str, default="../model/user_model/user_model.onnx")
    parser.add_argument("--do_export", type=bool, default=True, help="是否导出 ONNX 模型")
    args = parser.parse_args()

    model = UserModel(batch_size=args.batch_size, emb_slot_list=args.emb_slots)
    output_scores = model()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        feed_dict = {
            model.cross_emb_ph: np.random.rand(args.batch_size, args.emb_slots[0] * 64),
            model.other_emb_ph: np.random.rand(args.batch_size, args.emb_slots[1] * 64),
            model.pp_add_emb_ph: np.random.rand(args.batch_size, args.emb_slots[2] * 64),
            model.st_add_emb_ph: np.random.rand(args.batch_size, args.emb_slots[3] * 64),
            model.sc_add_emb_ph: np.random.rand(args.batch_size, args.emb_slots[4] * 64),
            model.lhuc_scene_emb_ph: np.random.rand(args.batch_size, args.emb_slots[5] * 64),
            model.tentacle_gate_emb_ph: np.random.rand(args.batch_size, args.emb_slots[6] * 64),
            model.dsif_gate_emb_ph: np.random.rand(args.batch_size, 32)
        }
        res = sess.run(output_scores, feed_dict=feed_dict)
        os.makedirs(os.path.dirname(args.ckpt_path), exist_ok=True)
        save_path = saver.save(sess, args.ckpt_path)
        print("✅ 模型保存至:", save_path)
        print("FD_CVR sample:", res['fd_cvr'][0])
        print("ST_CVR sample:", res['st_cvr'][0])

    if args.do_export:
        export_ckpt_to_onnx(args.ckpt_path, args.pb_path, args.onnx_path)
