

import numpy as np
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.ops.operations import TensorAdd
from mindspore import Tensor
from .lora_layer import LoRALinear
from .mobilenetV3 import MobileNetV2HeadV3Style
__all__ = ['MobileNetV2', 'MobileNetV2Backbone', 'MobileNetV2Head', 'mobilenet_v2']


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class GlobalAvgPooling(nn.Cell):

    def __init__(self):
        super(GlobalAvgPooling, self).__init__()
        self.mean = P.ReduceMean(keep_dims=False)

    def construct(self, x):
        x = self.mean(x, (2, 3))
        return x


class ConvBNReLU(nn.Cell):

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        super(ConvBNReLU, self).__init__()
        # 卷积边缘填充，使输出尺寸与输入保持一致（在 stride=1 时）
        padding = (kernel_size - 1) // 2
        # 保存输入/输出通道信息
        in_channels = in_planes
        out_channels = out_planes
        # 根据 groups 值判断是标准卷积还是深度卷积
        if groups == 1:
            # 标准卷积：groups=1，输出通道数由 out_planes 决定
            conv = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride,
                pad_mode='pad', padding=padding
            )
        else:
            # 深度卷积：groups=in_channels，每个输入通道独立卷积，不进行通道混合
            out_channels = in_planes  # 通道数保持不变
            conv = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride,
                pad_mode='pad', padding=padding, group=in_channels
            )
        # 构造卷积块：Conv → BN → ReLU6
        layers = [
            conv,
            nn.BatchNorm2d(out_planes),  # 批归一化
            nn.ReLU6()                   # 激活函数（限制最大值为 6，适合量化）
        ]

        # 使用 MindSpore 的 SequentialCell 打包成顺序网络模块
        self.features = nn.SequentialCell(layers)
    def construct(self, x):
        output = self.features(x)
        return output


class InvertedResidual(nn.Cell):

    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]  # stride 只能为 1 或 2，确保网络结构合法
        # 计算中间隐藏层的通道数：inp × expand_ratio
        hidden_dim = int(round(inp * expand_ratio))
        # 判断是否使用残差连接：
        # 条件：stride=1 且 输入输出通道数一致，才能直接加法相连
        self.use_res_connect = stride == 1 and inp == oup
        layers = []  # 用于保存各层结构
        #Expansion Layer：升维（如果 expand_ratio > 1）
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))  # 1×1 卷积升维
        #Depthwise Convolution：空间特征提取
        layers.extend([
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # DW卷积：组卷积中 groups=输入通道数，表示每个通道单独卷积
        ])
        # Projection Layer：降维（Linear Bottleneck，无激活）
        layers.extend([
            nn.Conv2d(hidden_dim, oup, kernel_size=1, stride=1, has_bias=False),  # 1x1 卷积降维
            nn.BatchNorm2d(oup),  # BN（无激活），线性输出
        ])
        # 将所有层用 SequentialCell 打包
        self.conv = nn.SequentialCell(layers)

        # 用于后续残差连接（x + identity）
        self.add = TensorAdd()
        self.cast = P.Cast()

    def construct(self, x):
        identity = x            # 保存输入，用于残差连接
        x = self.conv(x)        # 执行倒残差结构中的卷积堆叠

        if self.use_res_connect:
            return self.add(identity, x)  # 如果满足条件，执行残差连接
        return x                         # 否则，直接输出卷积结果（如降采样时）



class MobileNetV2Backbone(nn.Cell):
    def __init__(self, width_mult=1., inverted_residual_setting=None, round_nearest=8,
                 input_channel=32, last_channel=1280):
        super(MobileNetV2Backbone, self).__init__()

        # 倒残差模块类（一般为 InvertedResidual）
        block = InvertedResidual

        # 设置倒残差结构配置，如果外部没传，就使用默认标准配置
        self.cfgs = inverted_residual_setting
        if inverted_residual_setting is None:
            self.cfgs = [
                # [扩展倍数 t, 输出通道 c, 重复次数 n, 步长 s]
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # 根据缩放系数和对齐规则调整输入、输出通道数
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.out_channels = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)

        # 构建第一层卷积（3x3，stride=2）
        features = [ConvBNReLU(3, input_channel, stride=2)]

        # 构建 InvertedResidual 模块（堆叠多个 block）
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                # 第一次使用给定 stride，其余重复层 stride=1
                stride = s if i == 0 else 1
                # 添加一个 InvertedResidual 模块
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                # 更新输入通道为当前输出通道
                input_channel = output_channel
        # 添加最后一个 1x1 卷积层用于统一输出通道数
        features.append(ConvBNReLU(input_channel, self.out_channels, kernel_size=1))
        # 将所有层组合成一个顺序结构（类似 PyTorch 的 nn.Sequential）
        self.features = nn.SequentialCell(features)
        # 权重初始化
        self._initialize_weights()

    def construct(self, x):
        x = self.features(x)
        return x

    def _initialize_weights(self):
        self.init_parameters_data()
        for _, m in self.cells_and_names():
            if isinstance(m, nn.Conv2d):
                # He初始化：根据输出通道调整标准差
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.set_data(Tensor(np.random.normal(0, np.sqrt(2. / n),
                                                          m.weight.data.shape).astype("float32")))
                if m.bias is not None:
                    m.bias.set_data(Tensor(np.zeros(m.bias.data.shape, dtype="float32")))
            elif isinstance(m, nn.BatchNorm2d):
                m.gamma.set_data(Tensor(np.ones(m.gamma.data.shape, dtype="float32")))
                m.beta.set_data(Tensor(np.zeros(m.beta.data.shape, dtype="float32")))

    @property
    def get_features(self):
        return self.features


class MobileNetV2Head(nn.Cell):
    def __init__(self, input_channel=1280, num_classes=1000, has_dropout=False, activation="None", use_lora=False, r=4, alpha=16):
        super(MobileNetV2Head, self).__init__()

        self.need_activation = True
        if activation == "Sigmoid":
            self.activation = P.Sigmoid()
        elif activation == "Softmax":
            self.activation = P.Softmax()
        else:
            self.need_activation = False

        # 选择 Dense 或 LoRA 线性层
        if use_lora:
            dense_layer = LoRALinear(input_channel, num_classes, r=r, alpha=alpha)
        else:
            dense_layer = nn.Dense(input_channel, num_classes, has_bias=True)

        if not has_dropout:
            head = [
                GlobalAvgPooling(),
                dense_layer
            ]
        else:
            head = [
                GlobalAvgPooling(),
                nn.Dropout(0.2),
                dense_layer
            ]
        self.head = nn.SequentialCell(head)
        self._initialize_weights()


    def construct(self, x):
        x = self.head(x)  # 先经过全局池化 + 全连接等操作
        if self.need_activation:
            x = self.activation(x)  # 再根据配置执行激活函数
        return x

    def _initialize_weights(self):
        self.init_parameters_data()
        for _, m in self.cells_and_names():
            if isinstance(m, nn.Dense):
                m.weight.set_data(Tensor(np.random.normal(
                    0, 0.01, m.weight.data.shape).astype("float32")))
                if m.bias is not None:
                    m.bias.set_data(
                        Tensor(np.zeros(m.bias.data.shape, dtype="float32")))

    @property
    def get_head(self):
        """
        获取分类头结构，便于外部访问（如冻结、替换等）
        """
        return self.head


class MobileNetV2(nn.Cell):

    def __init__(self, num_classes=1000, width_mult=1., has_dropout=False,
                 inverted_residual_setting=None, round_nearest=8,
                 input_channel=32, last_channel=1280,
                 use_lora=False, lora_rank=4, lora_alpha=16):
        super(MobileNetV2, self).__init__()
        self.backbone = MobileNetV2Backbone(width_mult=width_mult, \
            inverted_residual_setting=inverted_residual_setting, \
            round_nearest=round_nearest, input_channel=input_channel, last_channel=last_channel).get_features
        # self.head = MobileNetV2Head(
        #     input_channel=self.backbone.out_channel,
        #     num_classes=num_classes,
        #     has_dropout=has_dropout,
        #     use_lora=use_lora,  # <-- 新增参数
        #     r=lora_rank,  # <-- 可调节的秩
        #     alpha=lora_alpha  # <-- 放大因子
        # ).get_head
        self.head = MobileNetV2HeadV3Style(
            input_channel=self.backbone.out_channel,
            num_classes=num_classes,
            has_dropout=has_dropout
        ).get_head

        # self.head = MobileNetV2Head(input_channel=self.backbone.out_channel, num_classes=num_classes, \
        #     has_dropout=has_dropout).get_head

    def construct(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x


class MobileNetV2Combine(nn.Cell):

    def __init__(self, backbone, head):
        super(MobileNetV2Combine, self).__init__(auto_prefix=False)
        self.backbone = backbone
        self.head = head

    def construct(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

def mobilenet_v2(backbone, head):
    return MobileNetV2Combine(backbone, head)
