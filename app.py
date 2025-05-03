import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import pandas as pd
import os
import gdown

# Path model
model_path = 'model3.h5'

# Cek apakah model sudah ada di lokal, jika tidak, download
if not os.path.exists(model_path):
    url = 'https://drive.google.com/uc?export=download&id=10R4hqfMd-QX3JahrGFat0Ts4XhBrnNkF'
    gdown.download(url, model_path, quiet=False)

# Load model
model = load_model(model_path)

# Label kelas sesuai encoding
class_labels = {
    0: 'Cars',
    1: 'Planes',
    2: 'Trains'
}

# Fungsi preprocessing gambar
def preprocess_image(image):
    image = image.resize((224, 224))  # Sesuaikan dengan input model
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Konfigurasi halaman Streamlit
st.set_page_config(page_title="Klasifikasi Gambar", layout="centered")
st.title("Klasifikasi Gambar: Cars, Planes, Trains")
st.write("Silakan upload gambar untuk diprediksi oleh model.")

# Upload gambar
uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang diunggah', use_container_width=True)

    if st.button("Prediksi"):
        with st.spinner("Memproses gambar..."):
            img_array = preprocess_image(image)
            prediction = model.predict(img_array)[0]
            pred_class_idx = np.argmax(prediction)
            pred_confidence = prediction[pred_class_idx] * 100

        st.subheader("Hasil Prediksi")

        if pred_confidence >= 97.0:
            pred_label = class_labels[pred_class_idx]
            st.write(f"Label: **{pred_label}**")
            st.write(f"Confidence: **{pred_confidence:.2f}%**")
        else:
            st.warning("Model tidak yakin. Gambar kemungkinan **bukan Cars, Planes, atau Trains**.")
            st.write(f"Confidence tertinggi: **{pred_confidence:.2f}%**")

        # Menampilkan skor confidence semua kelas
        st.subheader("Confidence Setiap Kelas")
        for i, prob in enumerate(prediction):
            label = class_labels[i]
            st.write(f"{label}: {prob * 100:.2f}%")

        # Visualisasi bar chart
        st.subheader("Visualisasi Confidence")
        df = pd.DataFrame({
            'Kelas': [class_labels[i] for i in range(len(prediction))],
            'Confidence': [round(p * 100, 2) for p in prediction]
        })
        st.bar_chart(df.set_index("Kelas"))
