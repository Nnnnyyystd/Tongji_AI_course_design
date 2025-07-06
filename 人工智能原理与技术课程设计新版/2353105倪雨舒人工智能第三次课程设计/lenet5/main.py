

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import mindspore
from mindspore import nn
from mindspore.dataset import vision, transforms, ImageFolderDataset
from mindspore import save_checkpoint, load_checkpoint, load_param_into_net

# 1. 数据路径配置
data_dir = "data"
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")

# 2. 数据预处理和加载（通道维修复）
def datapipe(data_dir, batch_size):
    dataset = ImageFolderDataset(data_dir, shuffle=True)

    image_transforms = [
        vision.Decode(),
        vision.ToPIL(),
        vision.Resize((28, 28)),
        vision.Grayscale(1),
        vision.Rescale(1.0 / 255.0, 0),
        vision.Normalize(mean=(0.1307,), std=(0.3081,)),
        vision.HWC2CHW(),
        lambda x: x.reshape(1, 28, 28)  # 👈 修复通道维，确保 shape 为 (1, 28, 28)
    ]
    label_transform = transforms.TypeCast(mindspore.int32)

    dataset = dataset.map(image_transforms, input_columns="image")
    dataset = dataset.map(label_transform, input_columns="label")
    dataset = dataset.batch(batch_size)
    return dataset

train_dataset = datapipe(train_dir, 64)
test_dataset = datapipe(test_dir, 64)

#3. 模型定义（LeNet5）
# class LeNet5(nn.Cell):
#     def __init__(self, num_classes=26):
#         super(LeNet5, self).__init__()
#         self.conv1 = nn.Conv2d(1, 6, kernel_size=5, pad_mode='valid')
#         self.relu1 = nn.ReLU()
#         self.pool1 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
#         self.conv2 = nn.Conv2d(6, 16, kernel_size=5, pad_mode='valid')
#         self.relu2 = nn.ReLU()
#         self.pool2 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Dense(16 * 4 * 4, 120)
#         self.fc2 = nn.Dense(120, 84)
#         self.fc3 = nn.Dense(84, num_classes)
#
#     def construct(self, x):
#         x = self.conv1(x)
#         x = self.relu1(x)
#         x = self.pool1(x)
#         x = self.conv2(x)
#         x = self.relu2(x)
#         x = self.pool2(x)
#         x = self.flatten(x)
#         x = self.fc1(x)
#         x = self.fc2(x)
#         x = self.fc3(x)
#         return x
class LeNet5(nn.Cell):
    def __init__(self, num_classes=26):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, pad_mode='valid')
        self.bn1 = nn.BatchNorm2d(6)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, pad_mode='valid')
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Dense(16 * 4 * 4, 84)  # ✅ 降低维度
        self.dropout1 = nn.Dropout(keep_prob=0.5)
        self.fc2 = nn.Dense(84, num_classes)

    def construct(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.flatten(x)
        x = self.dropout1(self.fc1(x))
        x = self.fc2(x)
        return x
model = LeNet5(num_classes=26)

# 4. 损失函数与优化器
loss_fn = nn.CrossEntropyLoss()
optimizer = nn.SGD(model.trainable_params(), 1e-2)

# 5. 前向 + 反向函数定义
def forward_fn(data, label):
    logits = model(data)
    loss = loss_fn(logits, label)
    return loss, logits

grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

def train_step(data, label):
    (loss, _), grads = grad_fn(data, label)
    optimizer(grads)
    return loss

# 6. 训练函数
def train(model, dataset):
    size = dataset.get_dataset_size()           # 获取训练集中 batch 总数（用于 tqdm 显示）
    model.set_train()                           # 设置模型为训练模式（启用 Dropout/BN 等）
    correct = 0                                  # 统计预测正确的样本数
    total = 0                                    # 样本总数
    total_loss = 0                               # 累加 loss
    for batch, (data, label) in enumerate(tqdm(dataset.create_tuple_iterator(), total=size, desc="Training")):
        loss = train_step(data, label)           # 执行一次 train_step（前向+反向+更新）
        total_loss += loss.asnumpy()             # 累加损失（转为 numpy 标量）
        logits = model(data)                     # 再前向一次用于评估准确率
        correct += (logits.argmax(1) == label).asnumpy().sum()  # argmax 取预测类别，与标签对比
        total += label.shape[0]                  # 样本总数
        if batch % 10 == 0:                      # 每 10 个 batch 输出一次中间 loss
            print(f"loss: {loss.asnumpy():>7f}  [{batch:>3d}/{size:>3d}]")

    avg_loss = total_loss / size                 # 平均每 batch 的损失
    accuracy = correct / total                   # 总体准确率
    print(f"Train: Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {avg_loss:>8f}\n")
    return avg_loss, accuracy                    # 返回给主循环记录

# 7. 测试函数
def test(model, dataset, loss_fn):
    num_batches = dataset.get_dataset_size()
    model.set_train(False)
    total, test_loss, correct = 0, 0, 0
    all_preds = []
    all_labels = []

    for data, label in dataset.create_tuple_iterator():
        pred = model(data)
        loss = loss_fn(pred, label)
        test_loss += loss.asnumpy()
        correct += (pred.argmax(1) == label).asnumpy().sum()
        total += label.shape[0]
        all_preds.extend(pred.argmax(1).asnumpy())
        all_labels.extend(label.asnumpy())

    test_loss /= num_batches
    accuracy = correct / total
    conf_matrix = confusion_matrix(all_labels, all_preds)
    print(f"Test: Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {test_loss:>8f}\n")
    return test_loss, accuracy, conf_matrix

#8. 主训练循环（含 EarlyStopping）
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []
final_conf_matrix = None

best_loss = float('inf')
early_stop_counter = 0
patience = 100  # 连续多少轮验证集 loss 未下降就提前终止

for t in range(1000):  # 最多训练 1000 轮
    print(f"Epoch {t+1}\n-------------------------------")
    train_loss, train_acc = train(model, train_dataset)
    test_loss, test_acc, conf_matrix = test(model, test_dataset, loss_fn)

    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)
    final_conf_matrix = conf_matrix

    # EarlyStopping 判断与模型保存
    if test_loss < best_loss:
        best_loss = test_loss
        early_stop_counter = 0
        save_checkpoint(model, "best_model.ckpt")
        print(" Best model updated.")
    else:
        early_stop_counter += 1
        print(f" No improvement in test loss for {early_stop_counter} epoch(s).")
        if early_stop_counter >= patience:
            print(f" Early stopping triggered at epoch {t+1}")
            break

#9. 保存混淆矩阵和模型
np.save("final_confusion_matrix.npy", final_conf_matrix)
with open("final_confusion_matrix.txt", "w") as f:
    f.write("Final Confusion Matrix:\n")
    for row in final_conf_matrix:
        f.write(" ".join(map(str, row)) + "\n")
print("混淆矩阵已保存")

save_checkpoint(model, "model.ckpt")
print("模型已保存到 model.ckpt")

# 10. 保存训练曲线
np.save("train_losses.npy", np.array(train_losses))
np.save("train_accuracies.npy", np.array(train_accuracies))
np.save("test_losses.npy", np.array(test_losses))
np.save("test_accuracies.npy", np.array(test_accuracies))

# 11. 模型加载测试

model = LeNet5(num_classes=26)
param_dict = load_checkpoint("best_model.ckpt")
load_param_into_net(model, param_dict)
model.set_train(False)

# 收集所有预测与真实标签
all_preds = []
all_labels = []

for data, label in test_dataset.create_tuple_iterator():
    pred = model(data)
    predicted = pred.argmax(1)
    all_preds.extend(predicted.asnumpy())
    all_labels.extend(label.asnumpy())

# 计算准确率
accuracy = accuracy_score(all_labels, all_preds)
print(f"✅ 最佳模型在验证集上的准确率为：{accuracy * 100:.2f}%")