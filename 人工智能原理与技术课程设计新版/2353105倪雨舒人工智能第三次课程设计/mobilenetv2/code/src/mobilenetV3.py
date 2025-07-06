import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore import Tensor
import numpy as np


class GlobalAvgPooling(nn.Cell):
    def __init__(self, keep_dims=True):
        super(GlobalAvgPooling, self).__init__()
        self.mean = P.ReduceMean(keep_dims=keep_dims)

    def construct(self, x):
        x = self.mean(x, (2, 3))  # 全局平均池化
        return x


class MobileNetV2HeadV3Style(nn.Cell):
    def __init__(self, input_channel=1280, mid_channel=1280, num_classes=1000,
                 has_dropout=True, activation="None"):
        super(MobileNetV2HeadV3Style, self).__init__()

        # 可选输出激活函数
        self.need_activation = True
        if activation == "Sigmoid":
            self.activation = P.Sigmoid()
        elif activation == "Softmax":
            self.activation = P.Softmax()
        else:
            self.need_activation = False

        layers = [
            GlobalAvgPooling(keep_dims=True),  # 输出为 [B, C, 1, 1]
            nn.Conv2d(in_channels=input_channel, out_channels=mid_channel,
                      kernel_size=1, has_bias=False, pad_mode='pad'),
            nn.BatchNorm2d(mid_channel),
            nn.HSwish()
        ]

        if has_dropout:
            layers.append(nn.Dropout(0.2))

        layers.append(nn.Conv2d(in_channels=mid_channel,
                                out_channels=num_classes,
                                kernel_size=1, has_bias=True, pad_mode='pad'))

        self.squeeze = P.Squeeze(axis=(2, 3))  # 去除 H, W
        self.head = nn.SequentialCell(layers)

        self._initialize_weights()

    def construct(self, x):
        x = self.head(x)
        x = self.squeeze(x)
        if self.need_activation:
            x = self.activation(x)
        return x

    def _initialize_weights(self):
        self.init_parameters_data()
        for _, m in self.cells_and_names():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.set_data(Tensor(np.random.normal(
                    0, np.sqrt(2. / n), m.weight.shape).astype("float32")))
                if m.bias is not None:
                    m.bias.set_data(Tensor(np.zeros(m.bias.shape, dtype="float32")))
            elif isinstance(m, nn.BatchNorm2d):
                m.gamma.set_data(Tensor(np.ones(m.gamma.shape, dtype="float32")))
                m.beta.set_data(Tensor(np.zeros(m.beta.shape, dtype="float32")))

    @property
    def get_head(self):
        return self.head
