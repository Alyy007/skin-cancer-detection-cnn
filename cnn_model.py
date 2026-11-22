
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# -------------------------------
# PAGE CONFIGURATION
# -------------------------------

st.set_page_config(
    page_title="Skin Cancer Detection",
    page_icon="🔬",
    layout="centered"
)

st.title("🔬 Skin Cancer Detection System")
st.write("Upload a skin image to predict whether it is Benign or Malignant.")

# -------------------------------
# MODEL PATH
# -------------------------------

MODEL_PATH = "skin_cancer_cnn_model.keras"

# -------------------------------
# LOAD TRAINED CNN MODEL
# -------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    st.error(
        "Model file not found! Please upload "
        "skin_cancer_cnn_model.keras to your GitHub repository."
    )
    st.stop()

try:
    model = load_model()
    st.success("CNN Model loaded successfully!")
except Exception as e:
    st.error("Error loading CNN model.")
    st.code(str(e))
    st.stop()

# -------------------------------
# IMAGE UPLOAD
# -------------------------------

uploaded_file = st.file_uploader(
    "Choose a Skin Image",
    type=["jpg", "jpeg", "png"]
)

# -------------------------------
# IMAGE PREPROCESSING
# -------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Skin Image",
        use_container_width=True
    )

    if st.button("Detect Skin Cancer", type="primary"):

        try:
            # Resize image to match CNN training size
            image = image.resize((128, 128))

            # Convert image into NumPy array
            image_array = np.array(image, dtype=np.float32)

            # Normalize image
            image_array = image_array / 255.0

            # Add batch dimension
            image_array = np.expand_dims(image_array, axis=0)

            # -------------------------------
            # CNN PREDICTION
            # -------------------------------

            prediction = model.predict(
                image_array,
                verbose=0
            )

            probability = float(prediction[0][0])

            # Original training class order:
            # benign = 0
            # malignant = 1

            if probability >= 0.5:
                result = "Malignant"
                confidence = probability * 100

            else:
                result = "Benign"
                confidence = (1 - probability) * 100

            # -------------------------------
            # DISPLAY RESULT
            # -------------------------------

            st.subheader("Prediction Result")

            if result == "Malignant":
                st.error("⚠️ Prediction: Malignant")
            else:
                st.success("Prediction: Benign")

            st.write(
                f"Confidence: {confidence:.2f}%"
            )

            st.progress(
                min(max(confidence / 100, 0.0), 1.0)
            )

            st.warning(
                "This application is for educational and research "
                "purposes only. It cannot diagnose skin cancer. "
                "Please consult a qualified dermatologist for "
                "medical assessment."
            )

        except Exception as e:
            st.error("Image prediction failed.")
            st.code(str(e))

else:
    st.info("Please upload a skin image to start detection.")

# -------------------------------
# FOOTER
# -------------------------------

st.divider()

st.caption(
    "Skin Cancer Detection using Convolutional Neural Network (CNN)"
)
