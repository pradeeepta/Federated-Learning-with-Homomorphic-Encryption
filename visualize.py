import numpy as np
import matplotlib.pyplot as plt
import os

# === CONFIG ===
num_clients = 6
disease_labels = ["Normal", "Lung Disease", "Pneumonia"]  # Adjust if needed

def plot_distribution(client_id):
    prefix = f"hospital{(client_id // 2) + 1}_{'lung' if client_id % 2 else 'pneumonia'}_client_{client_id}"
    train_labels_path = f"{prefix}_train_labels.npy"

    if not os.path.exists(train_labels_path):
        print(f"⚠️ Missing: {train_labels_path}")
        return

    labels = np.load(train_labels_path)
    unique, counts = np.unique(labels, return_counts=True)

    plt.bar([disease_labels[i] for i in unique], counts)
    plt.title(f"Client {client_id}: {prefix}")
    plt.ylabel("Samples")
    plt.xticks(rotation=15)
    plt.grid(True)
    plt.show()

for cid in range(num_clients):
    plot_distribution(cid)
