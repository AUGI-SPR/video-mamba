import matplotlib.pyplot as plt
import re

with open("logs/112732.out", "r") as file:
    log_data = file.read()

# Parse the log data
epochs = []
train_losses = []
train_accs = []
test_accs = []

for line in log_data.strip().split("\n"):
    if line.startswith("[epoch"):
        epoch = int(re.search(r"\[epoch (\d+)\]", line).group(1))
        loss = float(re.search(r"epoch loss = ([\d\.]+)", line).group(1))
        acc = float(re.search(r"acc = ([\d\.]+)", line).group(1))
        epochs.append(epoch)
        train_losses.append(loss)
        train_accs.append(acc)
    elif line.startswith("---[epoch"):
        tst_acc = float(re.search(r"tst acc = ([\d\.]+)", line).group(1))
        test_accs.append(tst_acc)


# 그래프 그리기
plt.figure(figsize=(14, 6))

# 훈련 손실 그래프
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label="Train Loss", color="blue")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()
plt.xlim(0, 120)  # x 축을 120까지만 설정

# 훈련 및 테스트 정확도 그래프
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accs, label="Train Accuracy", color="green")
plt.plot(epochs, test_accs, label="Test Accuracy", color="red")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training and Test Accuracy")
plt.legend()
plt.xlim(0, 120)  # x 축을 120까지만 설정

plt.tight_layout()

# 그래프를 PNG 파일로 저장
plt.savefig("training_results.png")

# 그래프 보여주기 (필요한 경우)
plt.show()
