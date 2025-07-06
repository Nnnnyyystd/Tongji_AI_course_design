
import os
import shutil
import numpy as np
from mindspore import Tensor, nn
from mindspore.train.model import Model
from mindspore.common import dtype as mstype

from src.dataset import create_dataset
from src.config import set_config
from src.args import eval_parse_args
from src.models import define_net, load_ckpt
from src.utils import switch_precision, set_context

from PIL import Image

if __name__ == '__main__':
    args_opt = eval_parse_args()
    config = set_config(args_opt)
    set_context(config)

    # 网络结构
    backbone_net, head_net, net = define_net(config, args_opt.is_training)
    load_ckpt(net, args_opt.pretrain_ckpt)
    switch_precision(net, mstype.float16, config)
    net.set_train(False)

    # 创建验证数据集
    dataset = create_dataset(dataset_path=args_opt.dataset_path, do_train=False, config=config)
    step_size = dataset.get_dataset_size()
    if step_size == 0:
        raise ValueError("The step_size of dataset is zero. Check if the images count of eval dataset is more \
            than batch_size in config.py")

    # 定义 loss 与模型
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    model = Model(net, loss_fn=loss)

    # 保存错误样本文件夹
    wrong_dir = './wrong_samples'
    if os.path.exists(wrong_dir):
        shutil.rmtree(wrong_dir)
    os.makedirs(wrong_dir)

    # 手动推理与评估
    total = 0
    correct = 0
    for data in dataset.create_dict_iterator():
        images = data["image"]
        labels = data["label"]
        outputs = net(images)
        preds = outputs.asnumpy().argmax(axis=1)
        labels_np = labels.asnumpy()

        for i in range(len(labels_np)):
            total += 1
            if preds[i] == labels_np[i]:
                correct += 1
            else:
                # 保存错误图像
                wrong_path = os.path.join(wrong_dir, f"wrong_{total}_pred{preds[i]}_label{labels_np[i]}.png")
                img_array = images[i].asnumpy().transpose(1, 2, 0) * 255  # CHW -> HWC
                img_array = np.clip(img_array, 0, 255).astype(np.uint8)
                img = Image.fromarray(img_array)
                img.save(wrong_path)

    acc = correct / total
    print(f"\n[Eval Finished] Total: {total}, Correct: {correct}, Acc: {acc:.4f}")
    print(f"[Wrong Samples Saved to]: {wrong_dir}")
