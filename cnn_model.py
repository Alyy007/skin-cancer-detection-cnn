    )
    st.stop()

try:
    model = load_cnn_model(str(MODEL_PATH))
except Exception as exc:
    st.error("The model file exists, but it could not be loaded.")
    st.code(str(exc))
    st.info(
        "Check that the .keras file is the model saved by your training code "
        "and that the TensorFlow version is compatible."
    )
    st.stop()

uploaded_file = st.file_uploader(
    "Choose a skin image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)

        # Match the training preprocessing: resize to 128x128 and divide by 255.
        resized_image = image.resize(IMG_SIZE)
        image_array = np.asarray(resized_image, dtype=np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        if st.button("Predict", type="primary"):
            with st.spinner("Analyzing image..."):
                probability = float(model.predict(image_array, verbose=0)[0][0])

            # The original training code uses flow_from_directory with
            # folders named benign and malignant. Alphabetical class order:
            # benign = 0, malignant = 1.
            if probability >= 0.5:
                predicted_class = "Malignant"
                confidence = probability * 100
            else:
                predicted_class = "Benign"
                confidence = (1.0 - probability) * 100

            st.subheader("Prediction")
            st.write(f"**Predicted class:** {predicted_class}")
            st.write(f"**Model confidence:** {confidence:.2f}%")

            st.warning(
                "Educational/research use only. This CNN prediction is not a "
                "medical diagnosis. Please consult a qualified dermatologist "
                "for assessment of any concerning skin lesion."
            )
    except Exception as exc:
        st.error("Could not process this image.")
        st.code(str(exc))
else:
    st.caption("Supported image formats: JPG, JPEG, PNG.")
