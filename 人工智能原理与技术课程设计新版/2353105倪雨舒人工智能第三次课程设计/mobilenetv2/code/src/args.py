

import argparse
import ast

def launch_parse_args():

    launch_parser = argparse.ArgumentParser(description="mindspore distributed training launch helper utilty \
        that will spawn up multiple distributed processes")
    launch_parser.add_argument('--platform', type=str, default="Ascend", choices=("Ascend", "GPU", "CPU"), \
        help='run platform, only support GPU, CPU and Ascend')
    launch_parser.add_argument("--nproc_per_node", type=int, default=1, choices=(1, 2, 3, 4, 5, 6, 7, 8), \
        help="The number of processes to launch on each node, for D training, this is recommended to be set \
            to the number of D in your system so that each process can be bound to a single D.")
    launch_parser.add_argument("--visible_devices", type=str, default="0,1,2,3,4,5,6,7", help="will use the \
        visible devices sequentially")
    launch_parser.add_argument("--training_script", type=str, default="./train.py", help="The full path to \
        the single D training program/script to be launched in parallel, followed by all the arguments for \
            the training script")

    launch_args, unknown = launch_parser.parse_known_args()
    launch_args.training_script_args = unknown
    launch_args.training_script_args += ["--platform", launch_args.platform]
    return launch_args

def train_parse_args():
    train_parser = argparse.ArgumentParser(description='Image classification train')

    train_parser.add_argument('--platform', type=str, default="CPU", choices=("CPU", "GPU", "Ascend"),
                              help='run platform, only support CPU, GPU and Ascend')

    #  数据集路径（根据你的项目结构）
    train_parser.add_argument('--dataset_path', type=str,
                              default="../data/train",
                              help='Dataset path')

    #  预训练模型路径（注意 ckpt 路径）
    train_parser.add_argument('--pretrain_ckpt', type=str,
                              default="../pretrain_checkpoint/mobilenetv2_cpu_gpu.ckpt",
                              help='Pretrained checkpoint path for fine tune or incremental learning')

    #  设置为冻结 backbone（或改为 "none" 做全量微调）
    train_parser.add_argument('--freeze_layer', type=str,
                              default="backbone", choices=["", "none", "backbone", "partial"],
                              help="freeze policy: 'none'=all trainable, 'backbone'=freeze all backbone, 'partial'=unfreeze last few layers")

    train_parser.add_argument('--run_distribute', type=ast.literal_eval,
                              default=False, help='Run distribute')

    #  LoRA 可选参数
    train_parser.add_argument('--use_lora', type=bool, default=False,
                              help='Enable LoRA in classification head')
    train_parser.add_argument('--lora_rank', type=int, default=8,
                              help='LoRA rank (low-rank dim)')
    train_parser.add_argument('--lora_alpha', type=float, default=16.0,
                              help='LoRA scaling factor')

    train_args = train_parser.parse_args()
    train_args.is_training = True
    return train_args


def eval_parse_args():
    eval_parser = argparse.ArgumentParser(description='Image classification eval')

    #  默认使用 CPU 测试，匹配训练配置
    eval_parser.add_argument('--platform', type=str, default="CPU", choices=("Ascend", "GPU", "CPU"),
                             help='run platform, only support GPU, CPU and Ascend')

    #  测试集路径（你的项目中有 data 文件夹）
    eval_parser.add_argument('--dataset_path', type=str,
                             default="../data/test",
                             help='Dataset path')

    #  指定微调后生成的模型 ckpt 路径（你已有 ckpt_0 文件夹）
    eval_parser.add_argument('--pretrain_ckpt', type=str,
                             default="../code/ckpt_0/mobilenetv2_30.ckpt",
                             help='Pretrained checkpoint path for fine tune or incremental learning')

    eval_parser.add_argument('--run_distribute', type=ast.literal_eval,
                             default=False, help='If run distribute in GPU.')

    #  LoRA 参数（如果未使用可保持默认）
    eval_parser.add_argument('--use_lora', type=bool, default=False, help='Whether to use LoRA structure')
    eval_parser.add_argument('--lora_rank', type=int, default=8, help='LoRA rank')
    eval_parser.add_argument('--lora_alpha', type=float, default=16.0, help='LoRA alpha scaling')

    eval_args = eval_parser.parse_args()
    eval_args.is_training = False
    return eval_args
