from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

MODEL_PATH = Path("ham10000_cnn_model.keras")

# ---- Settings: change these if your training used something different ----
CLASS_0 = "Benign"      # model output close to 0
CLASS_1 = "Malignant"   # model output close to 1
DIVIDE_BY_255 = True    # set False if you did NOT divide images by 255 in training
# --------------------------------------------------------------------------

st.set_page_config(page_title="Skin Lesion Classifier", page_icon="🩺")


@st.cache_resource
def load_model():
    import tensorflow as tf

    return tf.keras.models.load_model(MODEL_PATH, compile=False)


st.title("🩺 Skin Lesion Classifier (Benign vs Malignant)")
st.warning(
    "Educational prototype only. This is NOT a medical device and cannot diagnose "
    "skin cancer. See a dermatologist for any skin concern."
)

if not MODEL_PATH.exists():
    st.error(
        f"Model file `{MODEL_PATH}` not found. Upload it to the root of your GitHub "
        "repository (same folder as app.py) and reboot the app."
    )
    st.stop()

model = load_model()
height, width = model.input_shape[1:3]

uploaded = st.file_uploader("Upload a skin lesion image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    arr = np.array(image.resize((width, height)), dtype="float32")
    if DIVIDE_BY_255:
        arr = arr / 255.0
    p1 = float(model.predict(np.expand_dims(arr, 0), verbose=0)[0][0])

    label = CLASS_1 if p1 >= 0.5 else CLASS_0
    confidence = p1 if p1 >= 0.5 else 1 - p1
    st.subheader(f"Prediction: {label}")
    st.write(f"Confidence: {confidence:.1%}")
    st.bar_chart({CLASS_0: 1 - p1, CLASS_1: p1})
