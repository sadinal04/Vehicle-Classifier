import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import pandas as pd
import os
import gdown

# Konfigurasi halaman Streamlit (Pindah ke bagian paling atas)
st.set_page_config(page_title="Klasifikasi Gambar", layout="centered")

# Path model
model_path = 'model3.h5'

# Path untuk menyimpan model setelah diunduh
output_path = os.path.join("models", model_path)

# Pastikan folder 'models' ada
os.makedirs("models", exist_ok=True)

# Cek apakah model sudah ada di lokal, jika tidak, download
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
    st.write("Model berhasil dimuat.")
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# Label kelas sesuai encoding
class_labels = {
    0: 'Cars',
    1: 'Planes',
    2: 'Trains'
}

# Fungsi preprocessing gambar
def preprocess_image(image):
    image = image.resize((224, 224))  # Sesuaikan dengan input model
    img_array = np.array(image) / 255.0  # Normalisasi piksel gambar
    img_array = np.expand_dims(img_array, axis=0)  # Tambahkan dimensi batch
    return img_array

# Konfigurasi halaman Streamlit
st.title("Klasifikasi Gambar: Cars, Planes, Trains")
st.write("Silakan upload gambar untuk diprediksi oleh model.")

# Upload gambar
uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang diunggah', use_container_width=True)

    if st.button("Prediksi"):
        with st.spinner("Memproses gambar..."):
            try:
                # Proses gambar untuk prediksi
                img_array = preprocess_image(image)
                prediction = model.predict(img_array)[0]

                # Mendapatkan indeks kelas dengan probabilitas tertinggi
                pred_class_idx = np.argmax(prediction)
                pred_confidence = prediction[pred_class_idx] * 100

                # Menampilkan hasil prediksi
                st.subheader("Hasil Prediksi")
                if pred_confidence >= 97.0:
                    pred_label = class_labels[pred_class_idx]
                    st.write(f"Label: **{pred_label}**")
                    st.write(f"Confidence: **{pred_confidence:.2f}%**")
                else:
                    st.warning("Model tidak yakin. Gambar kemungkinan **bukan Cars, Planes, atau Trains**.")
                    st.write(f"Confidence tertinggi: **{pred_confidence:.2f}%**")

                # Menampilkan skor confidence untuk setiap kelas
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
                
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses gambar: {e}")
