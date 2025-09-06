import os
import numpy as np
import cv2
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

# =========================== CONFIG ===========================
image_size = (64, 64)
n_clients = 6
pca_components = 8
disease_types = ["lung", "pneumonia"]
base_path = "./"  # Folder containing hospital1/, hospital2/, ...
# =============================================================

def load_images_from_folder(folder_path, label):
    data = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img_resized = cv2.resize(img, image_size)
            data.append((img_resized.flatten(), label))
    return data

def prepare_data(hospital, disease):
    normal_path = os.path.join(hospital, disease, "normal")
    affected_path = os.path.join(hospital, disease, "affected")

    data = []
    data += load_images_from_folder(normal_path, 0)
    data += load_images_from_folder(affected_path, 1)
    print(f"✅ Loaded {len(data)} images for {hospital} - {disease}")

    random.shuffle(data)
    X = np.array([d[0] for d in data])
    y = np.array([d[1] for d in data])

    # Split for clients
    X_clients = np.array_split(X, n_clients)
    y_clients = np.array_split(y, n_clients)

    for i in range(n_clients):
        X_train, X_test, y_train, y_test = train_test_split(
            X_clients[i], y_clients[i], test_size=0.2, random_state=42, stratify=y_clients[i]
        )

        # Scale
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # SMOTE
        if len(np.unique(y_train)) > 1:
            sm = SMOTE(random_state=42)
            X_train, y_train = sm.fit_resample(X_train, y_train)

        # PCA
        pca = PCA(n_components=pca_components)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

        # Save
        prefix = f"{hospital}_{disease}_client_{i}"
        np.save(f"{prefix}_train_features.npy", X_train)
        np.save(f"{prefix}_train_labels.npy", y_train)
        np.save(f"{prefix}_test_features.npy", X_test)
        np.save(f"{prefix}_test_labels.npy", y_test)

        print(f"📁 Saved data for {prefix}")

if __name__ == "__main__":
    for hospital in ["hospital1", "hospital2", "hospital3"]:
        for disease in disease_types:
            prepare_data(hospital, disease)
