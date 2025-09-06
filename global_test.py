import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ========== MLP CLASSIFIER ==========
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(64, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return self.softmax(x)

# ========== DEVICE ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}")

# ========== LOAD GLOBAL MODEL ==========
model = MLPClassifier(input_dim=8, num_classes=3).to(device)
weights_list = torch.load("best_model.pt")  # it's a list of tensors, not a state_dict

with torch.no_grad():
    for param, w in zip(model.parameters(), weights_list):
        param.copy_(w.view(param.shape).to(torch.float32).to(device))

print("✅ Loaded global model from 'best_model.pt'")

# ========== LOAD & CONCATENATE TEST DATA ==========
all_test_features = []
all_test_labels = []

# Modify this list based on your actual setup
clients_info = [
    ("hospital1", "lung", 0),
    ("hospital2", "lung", 1),
    ("hospital3", "lung", 2),
    ("hospital1", "pneumonia", 3),
    ("hospital2", "pneumonia", 4),
    ("hospital3", "pneumonia", 5),
]

for hospital, disease, cid in clients_info:
    prefix = f"{hospital}_{disease}_client_{cid}"
    try:
        features = np.load(f"{prefix}_test_features.npy")
        labels = np.load(f"{prefix}_test_labels.npy")
        all_test_features.append(features)
        all_test_labels.append(labels)
        print(f"📂 Loaded {prefix} | Samples: {len(labels)}")
    except FileNotFoundError:
        print(f"⚠️ Missing: {prefix}_test_features.npy or labels")

# Combine all test sets
X_test = np.vstack(all_test_features)
y_test = np.hstack(all_test_labels)

# ========== EVALUATE ==========
model.eval()
with torch.no_grad():
    x = torch.tensor(X_test, dtype=torch.float32).to(device)import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from PIL import Image
import torchvision.transforms as transforms

# ========== MODEL DEFINITION ==========
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(64, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return self.softmax(x)

# ========== CONFIG ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = 8          # Same as PCA output
num_classes = 3        # Update if you used different classes
model_path = "best_model.pt"
image_path = "sample.jpg"  # CHANGE this to your image

# ========== LOAD MODEL ==========
model = MLPClassifier(input_dim=input_dim, num_classes=num_classes).to(device)
weights = torch.load(model_path)
with torch.no_grad():
    for param, w in zip(model.parameters(), weights):
        param.copy_(w.view(param.shape).to(device))
model.eval()

# ========== IMAGE PREPROCESSING ==========
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=1),  # Force grayscale if needed
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

img = Image.open(image_path).convert("RGB")
x = transform(img).view(-1).numpy()  # Flatten for MLP

# ========== SCALING + PCA ==========
scaler = StandardScaler()
pca = PCA(n_components=input_dim)

# Load sample training data to fit scaler & PCA (REQUIRED)
sample_data = np.load("hospital1_lung_client_0_train_features.npy")
sample_data = scaler.fit_transform(sample_data)
pca.fit(sample_data)

x_scaled = scaler.transform([x])
x_pca = pca.transform(x_scaled)

# ========== PREDICTION ==========
x_tensor = torch.tensor(x_pca, dtype=torch.float32).to(device)
with torch.no_grad():
    output = model(x_tensor)
    pred = output.argmax(dim=1).item()

# ========== CLASS LABELS ==========
class_labels = {0: "Normal", 1: "Pneumonia", 2: "Lung Disease"}  # Customize
print(f"\n📷 Prediction for '{image_path}': {class_labels.get(pred, pred)}")

    y = torch.tensor(y_test, dtype=torch.long).to(device)
    outputs = model(x)
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(outputs, y).item()
    preds = outputs.argmax(dim=1)
    acc = accuracy_score(y.cpu().numpy(), preds.cpu().numpy())

print(f"\n🌐 Final Global Evaluation:")
print(f"🔍 Accuracy: {acc:.4f}")
print(f"📉 Loss: {loss:.4f}")
