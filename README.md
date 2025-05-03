# Proyek Klasifikasi Gambar: Cars, Planes, Trains

## 🧑‍🤝‍🧑 Anggota Tim
- Aulia Vika Rahman  
- Sadinal Mufti  
- M. Nabil Maulana  
- M. Habil Aswad  
- M. Ihsan Rizqullah Adfa  

## 📝 Deskripsi Proyek
Proyek ini merupakan aplikasi klasifikasi gambar berbasis web yang dibuat menggunakan **Streamlit** dan **TensorFlow**. Aplikasi ini dapat mengklasifikasikan gambar menjadi tiga kategori utama:
- Cars
- Planes
- Trains

Model yang digunakan dilatih sebelumnya dan akan otomatis diunduh saat aplikasi dijalankan pertama kali.

## 🚀 Instruksi Penerapan

### ✅ Cara Termudah: Akses Melalui Link
Anda dapat langsung menggunakan aplikasi kami tanpa instalasi apa pun.  
Klik link berikut untuk membuka aplikasi yang sudah kami deploy:

👉 [https://vehicle-classifier-3huqpeduhjiwkbjkmucjch.streamlit.app/](https://vehicle-classifier-3huqpeduhjiwkbjkmucjch.streamlit.app/)

### 🔧 ATAU: Jalankan Aplikasi Secara Lokal

Ikuti langkah-langkah berikut untuk menjalankan aplikasi di komputer Anda:

```bash
# 1. Clone Repository
git clone https://github.com/sadinal04/Vehicle-Classifier
cd Vehicle-Classifier

# 2. Buat Virtual Environment (Opsional tapi Disarankan)
python -m venv venv
source venv/bin/activate   # Untuk Linux/MacOS
# atau
venv\Scripts\activate      # Untuk Windows

# 3. Install Dependensi
pip install -r requirements.txt

# 4. Jalankan Aplikasi
streamlit run app.py
