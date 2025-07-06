import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# === 加载保存的数据 ===
train_losses = np.load("train_losses.npy")
test_losses = np.load("test_losses.npy")
train_accuracies = np.load("train_accuracies.npy")
test_accuracies = np.load("test_accuracies.npy")

epochs = np.arange(1, len(train_losses) + 1)

# === 绘制 Loss 曲线 ===
plt.figure(figsize=(10, 4))
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Testing Loss Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_curve.png")
plt.show()

# === 绘制 Accuracy 曲线 ===
plt.figure(figsize=(10, 4))
plt.plot(epochs, train_accuracies, label="Train Accuracy")
plt.plot(epochs, test_accuracies, label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Testing Accuracy Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("accuracy_curve.png")
plt.show()


# 加载混淆矩阵
conf_matrix = np.load("final_confusion_matrix.npy")

# 类别标签（假设为 26 个大写字母）
labels = [chr(i) for i in range(ord('A'), ord('Z')+1)]

plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)

plt.title("Confusion Matrix (Test Set)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("confusion_matrix_visual.png", dpi=300)
plt.show()
