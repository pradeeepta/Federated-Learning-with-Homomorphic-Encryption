import tkinter as tk
from tkinter import filedialog, messagebox
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt


# ========== 1. Load the Trained CNN Model ==========

cnn_model = load_model('best_model.h5')
    

# ========== 2. Define Prediction Function ==========
def predict_image():
    file_path = filedialog.askopenfilename(
        title="Select Chest X-ray Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )

    if not file_path:
        print("❌ No file selected.")
        return

    # Preprocess image
    img = load_img(file_path, target_size=(224, 224))  # Resize as per model's input
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict
    prediction = cnn_model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class] * 100

    class_names = ['Affected', 'Normal']  # Adjust if more classes
    predicted_label = class_names[predicted_class]

    # Show results
    print(f"\n✅ Prediction: {predicted_label} ({confidence:.2f}%)")
    messagebox.showinfo("Prediction Result", f"Predicted Class: {predicted_label}\nConfidence: {confidence:.2f}%")

    # Show image with prediction
    plt.imshow(load_img(file_path))
    plt.axis('off')
    plt.title(f"{predicted_label} ({confidence:.2f}%)")
    plt.show()


# ========== 3. GUI Trigger ==========
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()  # Hide root window
    predict_image()
