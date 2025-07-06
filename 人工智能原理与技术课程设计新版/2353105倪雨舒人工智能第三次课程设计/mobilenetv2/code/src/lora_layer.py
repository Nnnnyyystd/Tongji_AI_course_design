import mindspore.nn as nn
import mindspore


class LoRALinear(nn.Cell):
    def __init__(self, in_features, out_features, r=4, alpha=1.0):
        super(LoRALinear, self).__init__()
        self.r = r
        self.alpha = alpha
        self.scale = alpha / r if r > 0 else 1

        # 原始全连接层权重（被冻结，不训练）
        self.weight = nn.Dense(in_features, out_features, has_bias=False)
        self.weight.to_float(mindspore.float32)
        self.weight.weight.requires_grad = False  # 冻结主干参数

        # LoRA A 和 B 层：低秩适配模块
        self.lora_a = nn.Dense(in_features, r, has_bias=False)
        self.lora_b = nn.Dense(r, out_features, has_bias=False)

    def construct(self, x):
        # 原始权重部分 + LoRA 部分
        return self.weight(x) + self.scale * self.lora_b(self.lora_a(x))

