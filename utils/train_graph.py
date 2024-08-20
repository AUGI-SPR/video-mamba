import matplotlib.pyplot as plt

# Corrected data (ensuring both lists have the same number of elements)
epochs = list(range(1, 10))
train_acc = [
    0.148422,
    0.237326,
    0.339906,
    0.468219,
    0.556884,
    0.680162,
    0.784001,
    0.722817,
    0.760366,
]
test_acc = [
    0.148422,
    0.138683,
    0.417024,
    0.623296,
    0.732372,
    0.731399,
    0.791391,
    0.763342,
    0.718543,
]

# Plotting the graph
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_acc, label="Train Accuracy", marker="o")
plt.plot(epochs, test_acc, label="Test Accuracy", marker="o")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Train and Test Accuracy over Epochs")
plt.legend()
plt.grid(True)
plt.xticks(epochs)  # Ensure all epochs are shown
plt.show()
