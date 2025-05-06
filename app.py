import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import pandas as pd
import os
import gdown

# Konfigurasi halaman
st.set_page_config(page_title="Klasifikasi Gambar", layout="centered")

# Inisialisasi session state
if 'use_camera' not in st.session_state:
    st.session_state.use_camera = False
if 'camera_image' not in st.session_state:
    st.session_state.camera_image = None

# Path model
model_path = 'model3.h5'
output_path = os.path.join("models", model_path)
os.makedirs("models", exist_ok=True)

# Unduh model jika belum tersedia
if not os.path.exists(output_path):
    url = 'https://drive.google.com/uc?export=download&id=10R4hqfMd-QX3JahrGFat0Ts4XhBrnNkF'
    st.write("Model belum ada di lokal. Mengunduh...")
    try:
        gdown.download(url, output_path, quiet=False)
    except Exception as e:
        st.error(f"Error saat mengunduh model: {e}")
        st.stop()

# Load model
try:
    model = load_model(output_path)
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# Label kelas
class_labels = {
    0: 'Cars',
    1: 'Planes',
    2: 'Trains'
}

def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# UI
st.title("🚀 Klasifikasi Gambar: Cars, Planes, Trains")
st.markdown("Unggah gambar dari perangkat atau **aktifkan kamera** untuk mengambil gambar langsung.")

image = None

# Tombol kamera
if st.button("📸 Ambil Gambar dari Kamera"):
    st.session_state.use_camera = True

# Tampilkan kamera hanya jika tombol ditekan
if st.session_state.use_camera:
    st.info("Silakan ambil gambar menggunakan kamera:")
    st.session_state.camera_image = st.camera_input("Klik tombol kamera di bawah untuk mengambil gambar")

# Upload file gambar
uploaded_file = st.file_uploader(
    "📂 Atau upload gambar dari perangkat",
    type=["jpg", "jpeg", "png"],
    label_visibility="visible"
)

# Ambil gambar dari kamera jika tersedia
if st.session_state.camera_image is not None:
    image = Image.open(st.session_state.camera_image)
    st.image(image, caption='Gambar dari Kamera', use_container_width=True)
elif uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang Diunggah', use_container_width=True)

# Prediksi
if image is not None:
    if st.button("🔍 Prediksi"):
        with st.spinner("Memproses gambar..."):
            try:
                img_array = preprocess_image(image)
                prediction = model.predict(img_array)[0]

                pred_class_idx = np.argmax(prediction)
                pred_confidence = prediction[pred_class_idx] * 100

                st.subheader("🧠 Hasil Prediksi")
                if pred_confidence >= 97.0:
                    pred_label = class_labels[pred_class_idx]
                    st.success(f"Label: **{pred_label}**")
                    st.write(f"Confidence: **{pred_confidence:.2f}%**")
                else:
                    st.warning("Model tidak yakin. Gambar kemungkinan **bukan Cars, Planes, atau Trains**.")
                    st.write(f"Confidence tertinggi: **{pred_confidence:.2f}%**")

                # Confidence setiap kelas
                st.subheader("📊 Confidence Setiap Kelas")
                for i, prob in enumerate(prediction):
                    label = class_labels[i]
                    st.write(f"{label}: {prob * 100:.2f}%")

                # Bar chart
                st.subheader("📈 Visualisasi Confidence")
                df = pd.DataFrame({
                    'Kelas': [class_labels[i] for i in range(len(prediction))],
                    'Confidence': [round(p * 100, 2) for p in prediction]
                })
                st.bar_chart(df.set_index("Kelas"))
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses gambar: {e}")
