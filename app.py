
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Fungsi untuk memuat model dan scaler
# Menggunakan cache untuk performa lebih baik, jadi model tidak di-load ulang setiap ada interaksi
@st.cache_resource
def load_model_and_scaler():
    """Memuat model dan scaler yang sudah dilatih."""
    model_path = os.path.join("outputs", "best_model.pkl")
    scaler_path = os.path.join("outputs", "scaler.pkl")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        st.error(
            "File model `best_model.pkl` atau `scaler.pkl` tidak ditemukan. "
            "Harap jalankan `main.py` terlebih dahulu untuk menghasilkan file-file ini."
        )
        return None, None

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

# Memuat model dan scaler
model, scaler = load_model_and_scaler()

# Judul dan deskripsi aplikasi
st.set_page_config(page_title="Prediksi Diabetes", layout="centered")
st.title("Aplikasi Web Prediksi Diabetes")
st.write(
    "Aplikasi ini menggunakan model Machine Learning untuk memprediksi risiko diabetes "
    "berdasarkan data medis pasien. Harap masukkan data pasien di bawah ini."
)

# Menampilkan informasi model jika berhasil dimuat
if model:
    st.info(
        "Model terbaik yang digunakan: **LightGBM** dengan fitur hasil seleksi **RFE**."
    )

# Form input data pasien di sidebar
with st.sidebar:
    st.header("Input Data Pasien")
    
    # Berdasarkan fitur yang dipilih oleh RFE di main.py:
    # ['Glucose', 'BloodPressure', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    # Kita perlu semua fitur original untuk proses scaling
    
    pregnancies = st.number_input("Jumlah Kehamilan (Pregnancies)", min_value=0, max_value=20, value=3, step=1)
    glucose = st.number_input("Kadar Glukosa (Glucose)", min_value=0, max_value=200, value=120, help="Konsentrasi glukosa plasma 2 jam dalam tes toleransi glukosa oral.")
    blood_pressure = st.number_input("Tekanan Darah (BloodPressure)", min_value=0, max_value=122, value=70, help="Tekanan darah diastolik (mm Hg).")
    skin_thickness = st.number_input("Ketebalan Kulit (SkinThickness)", min_value=0, max_value=99, value=20, help="Ketebalan lipatan kulit trisep (mm).")
    insulin = st.number_input("Kadar Insulin", min_value=0, max_value=846, value=80, help="Insulin serum 2 jam (mu U/ml).")
    bmi = st.number_input("Indeks Massa Tubuh (BMI)", min_value=0.0, max_value=67.1, value=32.0, format="%.1f", help="Berat badan dalam kg/(tinggi badan dalam m)^2.")
    dpf = st.number_input("Fungsi Silsilah Diabetes (DiabetesPedigreeFunction)", min_value=0.0, max_value=2.5, value=0.47, format="%.3f", help="Fungsi yang menilai kemungkinan diabetes berdasarkan riwayat keluarga.")
    age = st.number_input("Usia (Age)", min_value=21, max_value=85, value=33)

# Tombol untuk melakukan prediksi
predict_button = st.button("Prediksi Risiko Diabetes", type="primary")

# Proses prediksi ketika tombol ditekan
if model and scaler and predict_button:
    # Mengumpulkan semua fitur input ke dalam DataFrame, karena scaler dilatih pada semua fitur
    feature_names = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ]
    
    input_data = pd.DataFrame([[
        pregnancies, glucose, blood_pressure, skin_thickness, 
        insulin, bmi, dpf, age
    ]], columns=feature_names)

    # Scaling semua fitur
    input_data_scaled = scaler.transform(input_data)
    
    # Memilih hanya fitur yang digunakan oleh model RFE
    # Indeks RFE: [1, 2, 5, 6, 7] -> Glucose, BloodPressure, BMI, DiabetesPedigreeFunction, Age
    rfe_feature_indices = [1, 2, 5, 6, 7]
    final_input = input_data_scaled[:, rfe_feature_indices]

    # Melakukan prediksi
    try:
        prediction = model.predict(final_input)
        prediction_proba = model.predict_proba(final_input)

        st.divider()
        st.header("Hasil Prediksi")

        # Menampilkan hasil
        if prediction[0] == 1:
            st.warning(f"**Pasien Berisiko Terkena Diabetes** (Probabilitas: {prediction_proba[0][1]*100:.2f}%)")
            st.write("Disarankan untuk melakukan konsultasi lebih lanjut dengan dokter untuk pemeriksaan dan penanganan.")
        else:
            st.success(f"**Pasien Tidak Berisiko Terkena Diabetes** (Probabilitas: {prediction_proba[0][0]*100:.2f}%)")
            st.write("Tetap pertahankan gaya hidup sehat untuk pencegahan.")
            
    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")

# Footer
st.markdown("---")
st.write("Dibuat dengan Streamlit | Model ML oleh Pengguna")
