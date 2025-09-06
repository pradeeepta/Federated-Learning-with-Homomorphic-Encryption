import os
import torch
import torch.nn as nn
import numpy as np
from tkinter import filedialog, Tk, messagebox
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torchvision.transforms as transforms

# ========================= MLP MODEL =========================
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

# ========================= CONFIG =========================
input_dim = 8
num_classes = 3
model_path = "best_model.pt"
feature_path = "hospital1_lung_client_0_train_raw_features.npy"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================= LOAD MODEL =========================
model = MLPClassifier(input_dim=input_dim, num_classes=num_classes).to(device)
weights = torch.load(model_path)
with torch.no_grad():
    for param, w in zip(model.parameters(), weights):
        param.copy_(w.view(param.shape).to(device))
model.eval()

# ========================= TRANSFORM =========================
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ========================= LOAD FEATURE DATA =========================
if not os.path.exists(feature_path):
    print(f"❌ Feature file not found: {feature_path}")
    exit()

raw_features = np.load(feature_path)
scaler = StandardScaler()
pca = PCA(n_components=input_dim)
scaled = scaler.fit_transform(raw_features)
pca.fit(scaled)

# ========================= GUI FILE PICKER =========================
root = Tk()
root.withdraw()
file_path = filedialog.askopenfilename(title="📂 Please choose an image to predict:")

if not file_path:
    messagebox.showerror("Error", "No image selected!")
    exit()

# ========================= PROCESS IMAGE =========================
try:
    img = Image.open(file_path).convert("RGB")
    x = transform(img).view(-1).numpy()
    x_scaled = scaler.transform([x])
    x_pca = pca.transform(x_scaled)

    # Predict
    x_tensor = torch.tensor(x_pca, dtype=torch.float32).to(device)
    with torch.no_grad():
        output = model(x_tensor)
        pred = output.argmax(dim=1).item()

    # Label Mapping
    class_labels = {0: "Normal", 1: "Affected"}
    result = class_labels.get(pred, str(pred))

    messagebox.showinfo("Prediction Result", f"📷 Prediction for selected image:\n\n{result}")

except Exception as e:
    messagebox.showerror("Error", f"❌ Error processing image:\n{str(e)}")
