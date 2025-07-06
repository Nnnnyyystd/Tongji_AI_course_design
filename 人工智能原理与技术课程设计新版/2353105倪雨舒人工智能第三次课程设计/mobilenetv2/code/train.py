#
#
#
# import os
# import time
# import random
# import numpy as np
#
# from mindspore import Tensor
# from mindspore.nn import WithLossCell, TrainOneStepCell
# from mindspore.nn.optim.momentum import Momentum
# from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
# from mindspore.common import dtype as mstype
# from mindspore.communication.management import get_rank
# from mindspore.train.serialization import save_checkpoint
# from mindspore.common import set_seed
#
# # 自定义模块
# from src.dataset import extract_features
# from src.lr_generator import get_lr
# from src.config import set_config
# from src.args import train_parse_args
# from src.utils import context_device_init, switch_precision
# from src.models import CrossEntropyWithLabelSmooth, define_net, load_ckpt
# from mindspore.nn.optim import Adam
#
# # 固定随机种子，保证实验可复现
# set_seed(1)
#
# if __name__ == '__main__':
#     # 解析命令行参数
#     args_opt = train_parse_args()
#     config = set_config(args_opt)
#     # LoRA 参数转发到 config
#     config.use_lora = args_opt.use_lora
#     config.lora_rank = args_opt.lora_rank
#     config.lora_alpha = args_opt.lora_alpha
#
#     start = time.time()
#     print(f"train args: {args_opt}\ncfg: {config}")
#     # 初始化运行设备环境
#     context_device_init(config)
#     # 定义网络结构
#     backbone_net, head_net, net = define_net(config, args_opt.is_training)
#
#     # # 加载预训练的backbone并冻结其参数
#     # if args_opt.pretrain_ckpt != "":
#     #     load_ckpt(backbone_net, args_opt.pretrain_ckpt, trainable=False)
#     # else:
#     #     raise ValueError("Pretrained checkpoint required for fine-tuning head only.")
#
#     # 加载预训练的backbone并冻结其参数
#     if args_opt.pretrain_ckpt != "":
#         load_ckpt(backbone_net, args_opt.pretrain_ckpt, trainable=False)
#
#         # 冻结 backbone 的参数（确保只训练 head 或 LoRA）
#         for param in backbone_net.get_parameters():
#             param.requires_grad = False
#     else:
#         raise ValueError("Pretrained checkpoint required for fine-tuning head only.")
#
#     # 提取backbone特征保存为.npy格式
#     step_size = extract_features(backbone_net, args_opt.dataset_path, config)
#     if step_size == 0:
#         raise ValueError("No data found in dataset.")
#     # 启用 float16 加速（如平台支持）
#     switch_precision(net, mstype.float16, config)
#     # 构建损失函数
#     if config.label_smooth > 0:
#         loss = CrossEntropyWithLabelSmooth(
#             smooth_factor=config.label_smooth, num_classes=config.num_classes)
#     else:
#         loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
#
#     # 学习率调度
#     epoch_size = config.epoch_size
#     lr = Tensor(get_lr(global_step=0,
#                        lr_init=config.lr_init,
#                        lr_end=config.lr_end,
#                        lr_max=config.lr_max,
#                        warmup_epochs=config.warmup_epochs,
#                        total_epochs=epoch_size,
#                        steps_per_epoch=step_size))
#
#     # 只优化分类头 head_net 的参数
#     opt = Momentum(filter(lambda x: x.requires_grad, head_net.get_parameters()), lr, config.momentum, config.weight_decay)
#     #opt = Adam(filter(lambda x: x.requires_grad, head_net.get_parameters()),
#     #           learning_rate=lr,
#      #          weight_decay=config.weight_decay)
#
#     # 构建带 loss 的训练网络
#     network = WithLossCell(head_net, loss)
#     network = TrainOneStepCell(network, opt)
#     network.set_train()
#
#     # 加载特征路径
#     features_path = args_opt.dataset_path + '_features'
#     idx_list = list(range(step_size))
#     rank = 0
#     if config.run_distribute:
#         rank = get_rank()
#
#     # 创建 checkpoint 保存路径
#     save_ckpt_path = os.path.join(config.save_checkpoint_path, f'ckpt_{rank}/')
#     os.makedirs(save_ckpt_path, exist_ok=True)
#
#     # 开始训练
#     for epoch in range(epoch_size):
#         random.shuffle(idx_list)
#         epoch_start = time.time()
#         losses = []
#
#         for j in idx_list:
#             feature = Tensor(np.load(os.path.join(features_path, f"feature_{j}.npy")))
#             label = Tensor(np.load(os.path.join(features_path, f"label_{j}.npy")))
#             losses.append(network(feature, label).asnumpy())
#
#         epoch_mseconds = (time.time() - epoch_start) * 1000
#         per_step_mseconds = epoch_mseconds / step_size
#         print("epoch[{}/{}], iter[{}], cost: {:5.3f} ms, step: {:5.3f} ms, avg loss: {:5.3f}".format(
#             epoch + 1, epoch_size, step_size, epoch_mseconds, per_step_mseconds, np.mean(losses)))
#
#         if (epoch + 1) % config.save_checkpoint_epochs == 0:
#             save_checkpoint(net, os.path.join(save_ckpt_path, f"mobilenetv2_{epoch+1}.ckpt"))
#
#     print("total cost {:5.4f} s".format(time.time() - start))
import os
import time
import random
import numpy as np

from mindspore import Tensor
from mindspore.nn import WithLossCell, TrainOneStepCell
from mindspore.nn.optim.momentum import Momentum
from mindspore.nn.optim import Adam
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.common import dtype as mstype
from mindspore.communication.management import get_rank
from mindspore.train.serialization import save_checkpoint
from mindspore.common import set_seed

from src.dataset import extract_features, create_dataset
from src.lr_generator import get_lr
from src.config import set_config
from src.args import train_parse_args
from src.utils import context_device_init, switch_precision
from src.models import CrossEntropyWithLabelSmooth, define_net, load_ckpt
import psutil
import os
import time
import threading

def monitor_cpu_memory(log_file_path="cpu_mem_log.txt", interval=1.0, duration=None):
    """
    在后台线程中持续记录当前进程的 CPU 和内存使用情况。

    Args:
        log_file_path (str): 日志保存路径
        interval (float): 采样时间间隔（秒）
        duration (float or None): 总记录时间（秒），若为 None 则持续到主程序结束
    """
    process = psutil.Process(os.getpid())

    def log_loop():
        start_time = time.time()
        with open(log_file_path, "w") as f:
            f.write("Time,CPU_percent,Memory_MB\n")
            while duration is None or (time.time() - start_time) < duration:
                cpu = process.cpu_percent(interval=None)
                mem = process.memory_info().rss / (1024 * 1024)  # 转 MB
                f.write(f"{time.ctime()},{cpu:.2f},{mem:.2f}\n")
                f.flush()
                time.sleep(interval)

    # 创建并启动后台线程
    monitor_thread = threading.Thread(target=log_loop, daemon=True)
    monitor_thread.start()
set_seed(1)

def unfreeze_last_n_layers(backbone, unfreeze_start_idx=16):
    for name, param in backbone.parameters_and_names():
        if "features." in name:
            try:
                layer_num = int(name.split("features.")[1].split(".")[0])
                param.requires_grad = layer_num >= unfreeze_start_idx
            except ValueError:
                param.requires_grad = False
        else:
            param.requires_grad = False

if __name__ == '__main__':
    args_opt = train_parse_args()
    config = set_config(args_opt)
    config.use_lora = args_opt.use_lora
    config.lora_rank = args_opt.lora_rank
    config.lora_alpha = args_opt.lora_alpha

    start = time.time()
    print(f"train args: {args_opt}\ncfg: {config}")

    context_device_init(config)
    backbone_net, head_net, net = define_net(config, args_opt.is_training)

    is_full_finetune = args_opt.freeze_layer == "none"
    is_partial_finetune = args_opt.freeze_layer == "partial"
    is_backbone_frozen = args_opt.freeze_layer == "backbone"

    if args_opt.pretrain_ckpt:
        load_ckpt(backbone_net, args_opt.pretrain_ckpt, trainable=False)
        if is_partial_finetune:
            unfreeze_last_n_layers(backbone_net, unfreeze_start_idx=16)
        elif is_full_finetune:
            for param in backbone_net.get_parameters():
                param.requires_grad = True
        else:
            for param in backbone_net.get_parameters():
                param.requires_grad = False
    # else:
    #     raise ValueError("Pretrained checkpoint required for fine-tuning.")

    if is_backbone_frozen:
        step_size = extract_features(backbone_net, args_opt.dataset_path, config)
        if step_size == 0:
            raise ValueError("No data found in dataset.")
    else:
        dataset = create_dataset(dataset_path=args_opt.dataset_path, do_train=True, config=config)
        step_size = dataset.get_dataset_size()

    switch_precision(net, mstype.float16, config)

    if config.label_smooth > 0:
        loss = CrossEntropyWithLabelSmooth(smooth_factor=config.label_smooth, num_classes=config.num_classes)
    else:
        loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    epoch_size = config.epoch_size
    lr = Tensor(get_lr(global_step=0,
                       lr_init=config.lr_init,
                       lr_end=config.lr_end,
                       lr_max=config.lr_max,
                       warmup_epochs=config.warmup_epochs,
                       total_epochs=epoch_size,
                       steps_per_epoch=step_size))

    trainable_params = filter(lambda x: x.requires_grad, net.get_parameters() if not is_backbone_frozen else head_net.get_parameters())
    opt_cls = Adam if config.use_lora else Momentum #lora
    opt = opt_cls(trainable_params, learning_rate=lr, weight_decay=config.weight_decay) if config.use_lora \
        else Momentum(trainable_params, lr, config.momentum, config.weight_decay)

    if is_backbone_frozen:
        #monitor_cpu_memory(log_file_path="base_cpu_log.txt", interval=1.0)
        network = WithLossCell(head_net, loss)
        network = TrainOneStepCell(network, opt)
        network.set_train()
        features_path = args_opt.dataset_path + '_features'
        idx_list = list(range(step_size))
        rank = get_rank() if config.run_distribute else 0
        save_ckpt_path = os.path.join(config.save_checkpoint_path, f'ckpt_{rank}/')
        os.makedirs(save_ckpt_path, exist_ok=True)

        for epoch in range(epoch_size):
            random.shuffle(idx_list)
            epoch_start = time.time()
            losses = []
            for j in idx_list:
                feature = Tensor(np.load(os.path.join(features_path, f"feature_{j}.npy")))
                label = Tensor(np.load(os.path.join(features_path, f"label_{j}.npy")))
                losses.append(network(feature, label).asnumpy())
            epoch_milliseconds = (time.time() - epoch_start) * 1000
            print("epoch[{}/{}], avg loss: {:5.3f}, time: {:5.3f} ms".format(
                epoch + 1, epoch_size, np.mean(losses), epoch_milliseconds))
            if (epoch + 1) % config.save_checkpoint_epochs == 0:
                save_checkpoint(net, os.path.join(save_ckpt_path, f"mobilenetv2_{epoch+1}.ckpt"))
    else:
        from mindspore.train import Model
        from mindspore.train.loss_scale_manager import FixedLossScaleManager
        loss_scale = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
        model = Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale)
        from src.utils import config_ckpoint
        cb = config_ckpoint(config, lr, step_size)
        print("============== 开始训练 ==============")
        monitor_cpu_memory(log_file_path="full_cpu_log.txt", interval=1.0)
        model.train(epoch_size, dataset, callbacks=cb)
        print("============== 训练结束 ==============")

    print("Total time: {:5.4f}s".format(time.time() - start))
