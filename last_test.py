import torch
import numpy as np
import joblib
from PIL import Image
import cv2
import os
import tkinter as tk
from tkinter import filedialog

# ======= 1. Define the MLP Model =======
class MLPClassifier(torch.nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLPClassifier, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 128)
        self.relu = torch.nn.ReLU()
        self.dropout1 = torch.nn.Dropout(0.4)
        self.fc2 = torch.nn.Linear(128, 64)
        self.dropout2 = torch.nn.Dropout(0.4)
        self.fc3 = torch.nn.Linear(64, num_classes)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return self.softmax(x)

# ======= 2. Feature Extraction from Image =======
def extract_features(img_path):
    try:
        image = cv2.imread(img_path)
        image = cv2.resize(image, (64, 64))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # ⚠️ Only 8 histogram bins to match scaler input
        hist = cv2.calcHist([gray], [0], None, [8], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist
    except Exception as e:
        print(f"❌ Failed to extract features from {img_path}: {e}")
        return None


# ======= 3. File Dialog in place of argparse =======
root = tk.Tk()
root.withdraw()  # Hide main window

image_path = filedialog.askopenfilename(
    title="Select an image",
    filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
)

if not image_path:
    print("❌ No image selected. Exiting.")
    exit(1)

# ======= 4. Load Model and Preprocessors =======
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
weights = torch.load("best_model.pt")

model = MLPClassifier(input_dim=8, num_classes=3)
with torch.no_grad():
    for param, weight in zip(model.parameters(), weights):
        param.copy_(weight.view(param.shape))
model.eval()

# ======= 5. Process Image =======
features = extract_features(image_path)
if features is None:
    exit(1)

scaled = scaler.transform([features])
reduced = pca.transform(scaled)
input_tensor = torch.tensor(reduced, dtype=torch.float32)

# ======= 6. Predict =======
with torch.no_grad():
    output = model(input_tensor)
    predicted_class = output.argmax(dim=1).item()
    print(f"🧠 Predicted Class for {os.path.basename(image_path)}: {predicted_class}")
