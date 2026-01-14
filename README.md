🩺 Federated Learning for Pneumonia Detection with Privacy Preservation

This capstone project implements a Federated Learning (FL) framework for medical image classification using Chest X-Ray Pneumonia datasets, with a strong focus on data privacy and security.

Instead of sending sensitive medical data to a centralized server, multiple simulated clients train models locally on their own data. Only encrypted model updates are shared with the server, ensuring that patient data never leaves the client side.

🔍 Key Features

Federated learning setup with multiple simulated clients

MobileNetV2 pre-trained CNN for efficient and accurate image classification

Homomorphic encryption (TenSEAL / Paillier) applied to model updates to preserve privacy

Secure aggregation of encrypted weights on the server

End-to-end training and evaluation pipeline

Designed for healthcare use cases where data confidentiality is critical

🛠️ Tech Stack

Python

TensorFlow / Keras

Federated Learning (FL)

Homomorphic Encryption (TenSEAL / Paillier)

Chest X-Ray Pneumonia Dataset

Kaggle / Local Simulation Environment

🎯 Objective

The goal of this project is to demonstrate how privacy-preserving machine learning can be applied to real-world healthcare problems, enabling collaborative model training without compromising sensitive medical data.

📈 Results

The federated model achieves competitive accuracy while maintaining strict data privacy, proving that secure and decentralized learning is a viable approach for medical image analysis.
