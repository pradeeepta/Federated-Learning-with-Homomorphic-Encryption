

import torch
import numpy as np
import joblib
from PIL import Image
import cv2
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from tensorflow.keras.models import load_model as keras_load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

def predict():
   
    mlp_model = keras_load_model('best_model.h5')

    file_path = filedialog.askopenfilename(title="Select an image", filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    if not file_path:
        print("❌ No file selected.")
        return

    img = load_img(file_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = mlp_model.predict(img_array)
    predicted_class = np.argmax(prediction)
    class_names = ['Affected', 'Normal']

    predicted_label = class_names[predicted_class]
    confidence = prediction[0][predicted_class] * 100

    print(f"\n Prediction: {predicted_label}")


    messagebox.showinfo("MLP Prediction",
                        f"Predicted Class: {predicted_label}\nConfidence: {confidence:.2f}% (MLP Model)")

    plt.imshow(load_img(file_path))
    plt.axis('off')
    plt.title(f"MLP Prediction: {predicted_label} ({confidence:.2f}%)")
    plt.show()

# ======= 2. Main Entry =======
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    predict()
