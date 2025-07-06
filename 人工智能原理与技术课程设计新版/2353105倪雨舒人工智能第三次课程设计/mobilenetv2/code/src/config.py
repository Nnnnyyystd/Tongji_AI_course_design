
"""
network config setting, will be used in train.py and eval.py
"""
import os
from easydict import EasyDict as ed

def set_config(args):
    config_cpu = ed({
        # 数据相关
        "num_classes": 26,  # 垃圾分类类别数
        "image_height": 224,
        "image_width": 224,
        "batch_size": 32,

        # 训练控制
        "epoch_size": 100,
        "warmup_epochs": 0,

        # 学习率调度
        "lr_init": 0.0,
        "lr_end": 0.03,
        "lr_max": 0.03,

        # 优化器参数
        "momentum": 0.9,
        "weight_decay": 4e-5,

        # 损失函数设置
        "label_smooth": 0.1,

        # 数值精度与稳定性
        "loss_scale": 1024,

        # Checkpoint 设置
        "save_checkpoint": True,
        "save_checkpoint_epochs": 5,
        "keep_checkpoint_max": 20,
        "save_checkpoint_path": "./",

        # 系统运行设置
        "platform": args.platform,
        "run_distribute": False,

        # 网络结构控制
        "activation": "Softmax",

        #  LoRA 微调控制参数
        "use_lora": args.use_lora,  # ← 来自命令行参数
        "lora_rank": args.lora_rank,  # ← r 参数
        "lora_alpha": args.lora_alpha  # ← α 参数
    })

    config_gpu = ed({
        "num_classes": 1000,
        "image_height": 224,
        "image_width": 224,
        "batch_size": 150,
        "epoch_size": 200,
        "warmup_epochs": 0,
        "lr_init": .0,
        "lr_end": .0,
        "lr_max": 0.8,
        "momentum": 0.9,
        "weight_decay": 4e-5,
        "label_smooth": 0.1,
        "loss_scale": 1024,
        "save_checkpoint": True,
        "save_checkpoint_epochs": 1,
        "keep_checkpoint_max": 200,
        "save_checkpoint_path": "./",
        "platform": args.platform,
        "ccl": "nccl",
        "run_distribute": args.run_distribute,
        "activation": "Softmax"
    })
    config_ascend = ed({
        "num_classes": 1000,
        "image_height": 224,
        "image_width": 224,
        "batch_size": 256,
        "epoch_size": 200,
        "warmup_epochs": 4,
        "lr_init": 0.00,
        "lr_end": 0.00,
        "lr_max": 0.4,
        "momentum": 0.9,
        "weight_decay": 4e-5,
        "label_smooth": 0.1,
        "loss_scale": 1024,
        "save_checkpoint": True,
        "save_checkpoint_epochs": 1,
        "keep_checkpoint_max": 200,
        "save_checkpoint_path": "./",
        "platform": args.platform,
        "ccl": "hccl",
        "device_id": int(os.getenv('DEVICE_ID', '0')),
        "rank_id": int(os.getenv('RANK_ID', '0')),
        "rank_size": int(os.getenv('RANK_SIZE', '1')),
        "run_distribute": int(os.getenv('RANK_SIZE', '1')) > 1.,
        "activation": "Softmax"
    })
    config = ed({"CPU": config_cpu,
                 "GPU": config_gpu,
                 "Ascend": config_ascend})

    if args.platform not in config.keys():
        raise ValueError("Unsupport platform.")

    return config[args.platform]
