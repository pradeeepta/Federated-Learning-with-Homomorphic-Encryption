import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import torchvision.transforms as transforms
from glob import glob
from sklearn.utils import shuffle

# ========== SETTINGS ==========
input_dim = 8
num_clients = 6
image_size = (64, 64)
hospitals = ["hospital1", "hospital2", "hospital3"]
diseases = ["lung", "pneumonia"]
classes = ["normal", "affected"]

# ========== TRANSFORM ==========
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ========== LOOP ==========
for hospital in hospitals:
    for disease in diseases:
        print(f"\n🏥 Processing {hospital} - {disease}")
        
        all_features = []
        all_labels = []

        # Load images from both classes
        for label_idx, class_type in enumerate(classes):
            folder = os.path.join(hospital, disease, class_type)
            image_paths = glob(os.path.join(folder, "*.jpeg")) + \
                          glob(os.path.join(folder, "*.jpg")) + \
                          glob(os.path.join(folder, "*.png"))

            for img_path in image_paths:
                try:
                    img = Image.open(img_path).convert("RGB")
                    x = transform(img).view(-1).numpy()  # 4096D
                    all_features.append(x)
                    all_labels.append(label_idx)
                except Exception as e:
                    print(f"⚠️ Skipped {img_path}: {e}")

        all_features, all_labels = shuffle(all_features, all_labels, random_state=42)
        all_features = np.array(all_features)
        all_labels = np.array(all_labels)

        total_samples = len(all_features)
        samples_per_client = total_samples // num_clients

        for client_id in range(num_clients):
            start = client_id * samples_per_client
            end = (client_id + 1) * samples_per_client if client_id < num_clients - 1 else total_samples

            client_features = all_features[start:end]
            client_labels = all_labels[start:end]

            # Save raw 4096D features
            raw_file = f"{hospital}_{disease}_client_{client_id}_train_raw_features.npy"
            np.save(raw_file, client_features)
            print(f"📁 Saved raw features → {raw_file}")

            # Scale + PCA
            scaler = StandardScaler()
            scaled = scaler.fit_transform(client_features)

            pca = PCA(n_components=input_dim)
            reduced = pca.fit_transform(scaled)

            train_x, test_x = train_test_split(reduced, test_size=0.2, random_state=42)

            # Save reduced features
            np.save(f"{hospital}_{disease}_client_{client_id}_train_features.npy", train_x)
            np.save(f"{hospital}_{disease}_client_{client_id}_test_features.npy", test_x)

            print(f"✅ Client {client_id} - {hospital}-{disease} | Train: {train_x.shape} | Test: {test_x.shape}")
