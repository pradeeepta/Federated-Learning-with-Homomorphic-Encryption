from tensorflow.keras.models import load_model
import numpy as np

model = load_model("best_model.h5")
weights = model.get_weights()
real_weights = [w.astype(np.float32).tobytes() for w in weights]

log_model_update(
    hospital_id="manipal",
    weights_bytes=real_weights,
    accuracy=91.37,
    epoch="12"
)
