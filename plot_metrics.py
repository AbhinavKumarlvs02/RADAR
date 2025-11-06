import json
import matplotlib.pyplot as plt

HISTORY_PATH = "training_history.json"
PLOT_SAVE_PATH = "metrics_plot.png"

print(f"Loading history from: {HISTORY_PATH}")

try:
    with open(HISTORY_PATH, 'r') as f:
        history = json.load(f)
except FileNotFoundError:
    print(f"Error: {HISTORY_PATH} not found.")
    print("Please run train.py first to generate the history file.")
    exit()

# Extract data
train_loss = history['train_loss']
val_loss = history['val_loss']
val_acc = history['val_acc']
epochs = range(1, len(train_loss) + 1)

# --- Plot 1: Loss ---
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, 'bo-', label='Training Loss')
plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# --- Plot 2: Accuracy ---
plt.subplot(1, 2, 2)
plt.plot(epochs, val_acc, 'go-', label='Validation Accuracy')
plt.title('Validation Pixel Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.savefig(PLOT_SAVE_PATH)
print(f"Metrics plot saved to: {PLOT_SAVE_PATH}")

# Show the plot
plt.show()