#
# import time
# import numpy as np
# from mindspore import Tensor
# from mindspore import nn
# from mindspore.ops import operations as P
# from mindspore.ops import functional as F
# from mindspore.common import dtype as mstype
# from mindspore.nn.loss.loss import _Loss
# from mindspore.train.callback import Callback
# from mindspore.train.serialization import load_checkpoint, load_param_into_net
# from src.mobilenetV2 import MobileNetV2Backbone, MobileNetV2Head, mobilenet_v2
#
# class CrossEntropyWithLabelSmooth(_Loss):
#
#     def __init__(self, smooth_factor=0., num_classes=1000):
#         super(CrossEntropyWithLabelSmooth, self).__init__()
#         self.onehot = P.OneHot()
#         self.on_value = Tensor(1.0 - smooth_factor, mstype.float32)
#         self.off_value = Tensor(1.0 * smooth_factor /
#                                 (num_classes - 1), mstype.float32)
#         self.ce = nn.SoftmaxCrossEntropyWithLogits()
#         self.mean = P.ReduceMean(False)
#         self.cast = P.Cast()
#
#     def construct(self, logit, label):
#         one_hot_label = self.onehot(self.cast(label, mstype.int32), F.shape(logit)[1],
#                                     self.on_value, self.off_value)
#         out_loss = self.ce(logit, one_hot_label)
#         out_loss = self.mean(out_loss, 0)
#         return out_loss
#
# class Monitor(Callback):
#
#
#     def __init__(self, lr_init=None):
#         super(Monitor, self).__init__()
#         self.lr_init = lr_init
#         self.lr_init_len = len(lr_init)
#
#     def epoch_begin(self, run_context):
#         self.losses = []
#         self.epoch_time = time.time()
#
#     def epoch_end(self, run_context):
#         cb_params = run_context.original_args()
#
#         epoch_mseconds = (time.time() - self.epoch_time) * 1000
#         per_step_mseconds = epoch_mseconds / cb_params.batch_num
#         print("epoch time: {:5.3f}, per step time: {:5.3f}, avg loss: {:5.3f}".format(epoch_mseconds,
#                                                                                       per_step_mseconds,
#                                                                                       np.mean(self.losses)))
#
#     def step_begin(self, run_context):
#         self.step_time = time.time()
#
#     def step_end(self, run_context):
#         cb_params = run_context.original_args()
#         step_mseconds = (time.time() - self.step_time) * 1000
#         step_loss = cb_params.net_outputs
#
#         if isinstance(step_loss, (tuple, list)) and isinstance(step_loss[0], Tensor):
#             step_loss = step_loss[0]
#         if isinstance(step_loss, Tensor):
#             step_loss = np.mean(step_loss.asnumpy())
#
#         self.losses.append(step_loss)
#         cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num
#
#         print("epoch: [{:3d}/{:3d}], step:[{:5d}/{:5d}], loss:[{:5.3f}/{:5.3f}], time:[{:5.3f}], lr:[{:5.3f}]".format(
#             cb_params.cur_epoch_num -
#             1, cb_params.epoch_num, cur_step_in_epoch, cb_params.batch_num, step_loss,
#             np.mean(self.losses), step_mseconds, self.lr_init[cb_params.cur_step_num - 1]))
#
# def load_ckpt(network, pretrain_ckpt_path, trainable=True):
#     """
#     incremental_learning or not
#     """
#     param_dict = load_checkpoint(pretrain_ckpt_path)
#     load_param_into_net(network, param_dict)
#     if not trainable:
#         for param in network.get_parameters():
#             param.requires_grad = False
#
# def define_net(config, is_training):
#     backbone_net = MobileNetV2Backbone()
#     activation = config.activation if not is_training else "None"
#     head_net = MobileNetV2Head(input_channel=backbone_net.out_channels,
#                                num_classes=config.num_classes,
#                                activation=activation)
#     net = mobilenet_v2(backbone_net, head_net)
#     return backbone_net, head_net, net
import time
import numpy as np
from mindspore import Tensor
from mindspore import nn
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common import dtype as mstype
from mindspore.nn.loss.loss import _Loss
from mindspore.train.callback import Callback
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.mobilenetV2 import MobileNetV2Backbone, MobileNetV2Head, mobilenet_v2
from src.lora_layer import LoRALinear  # ✅ 新增导入（不直接用，但提醒依赖）

# ✅ 带标签平滑的交叉熵损失
class CrossEntropyWithLabelSmooth(_Loss):
    def __init__(self, smooth_factor=0., num_classes=1000):
        super(CrossEntropyWithLabelSmooth, self).__init__()
        self.onehot = P.OneHot()
        self.on_value = Tensor(1.0 - smooth_factor, mstype.float32)
        self.off_value = Tensor(1.0 * smooth_factor / (num_classes - 1), mstype.float32)
        self.ce = nn.SoftmaxCrossEntropyWithLogits()
        self.mean = P.ReduceMean(False)
        self.cast = P.Cast()

    def construct(self, logit, label):
        one_hot_label = self.onehot(self.cast(label, mstype.int32), F.shape(logit)[1],
                                    self.on_value, self.off_value)
        out_loss = self.ce(logit, one_hot_label)
        out_loss = self.mean(out_loss, 0)
        return out_loss

# ✅ 简单训练监控 Callback
class Monitor(Callback):
    def __init__(self, lr_init=None):
        super(Monitor, self).__init__()
        self.lr_init = lr_init
        self.lr_init_len = len(lr_init)

    def epoch_begin(self, run_context):
        self.losses = []
        self.epoch_time = time.time()

    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        epoch_mseconds = (time.time() - self.epoch_time) * 1000
        per_step_mseconds = epoch_mseconds / cb_params.batch_num
        print("epoch time: {:5.3f}, per step time: {:5.3f}, avg loss: {:5.3f}".format(
            epoch_mseconds, per_step_mseconds, np.mean(self.losses)))

    def step_begin(self, run_context):
        self.step_time = time.time()

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        step_mseconds = (time.time() - self.step_time) * 1000
        step_loss = cb_params.net_outputs

        if isinstance(step_loss, (tuple, list)) and isinstance(step_loss[0], Tensor):
            step_loss = step_loss[0]
        if isinstance(step_loss, Tensor):
            step_loss = np.mean(step_loss.asnumpy())

        self.losses.append(step_loss)
        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num

        print("epoch: [{:3d}/{:3d}], step:[{:5d}/{:5d}], loss:[{:5.3f}/{:5.3f}], time:[{:5.3f}], lr:[{:5.3f}]".format(
            cb_params.cur_epoch_num - 1, cb_params.epoch_num,
            cur_step_in_epoch, cb_params.batch_num, step_loss,
            np.mean(self.losses), step_mseconds,
            self.lr_init[cb_params.cur_step_num - 1]))

# ✅ 预训练权重加载函数（支持冻结）
def load_ckpt(network, pretrain_ckpt_path, trainable=True):
    param_dict = load_checkpoint(pretrain_ckpt_path)
    load_param_into_net(network, param_dict)
    if not trainable:
        for param in network.get_parameters():
            param.requires_grad = False

#  网络结构构建函数（支持 use_lora 控制）
def define_net(config, is_training):
    # Backbone: MobileNetV2 特征提取
    backbone_net = MobileNetV2Backbone()

    # 分类头激活函数（训练阶段一般不用激活）
    activation = config.activation if not is_training else "None"

    # Head: 分类头，支持是否使用 LoRA
    head_net = MobileNetV2Head(
        input_channel=backbone_net.out_channels,
        num_classes=config.num_classes,
        activation=activation,
        has_dropout=config.has_dropout if hasattr(config, "has_dropout") else False,
        use_lora=config.use_lora if hasattr(config, "use_lora") else False,
        r=config.lora_rank if hasattr(config, "lora_rank") else 4,
        alpha=config.lora_alpha if hasattr(config, "lora_alpha") else 16
    )

    # Combine 网络
    net = mobilenet_v2(backbone_net, head_net)
    return backbone_net, head_net, net
