
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt

# Fungsi untuk menampilkan plot SHAP di Streamlit
def st_shap(plot, height=None):
    """Menampilkan plot SHAP di dalam aplikasi Streamlit."""
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    st.components.v1.html(shap_html, height=height)

# Fungsi untuk memuat aset-aset yang diperlukan
@st.cache_resource
def load_assets():
    """Memuat model, scaler, SHAP explainer, dan metrik performa."""
    model_path = os.path.join("outputs", "best_model.pkl")
    scaler_path = os.path.join("outputs", "scaler.pkl")
    results_path = os.path.join("outputs", "summary_results.csv")

    if not all(os.path.exists(p) for p in [model_path, scaler_path, results_path]):
        st.error(
            "Satu atau lebih file aset (`best_model.pkl`, `scaler.pkl`, `summary_results.csv`) tidak ditemukan. "
            "Harap jalankan `main.py` terlebih dahulu untuk menghasilkan file-file ini."
        )
        return None, None, None, None

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    explainer = shap.TreeExplainer(model)
    
    # Memuat hasil dan mencari metrik model terbaik
    results_df = pd.read_csv(results_path)
    best_model_metrics = results_df.loc[results_df['f1'].idxmax()]
    
    return model, scaler, explainer, best_model_metrics

# Memuat aset
model, scaler, explainer, metrics = load_assets()

# Konfigurasi halaman dan judul
st.set_page_config(page_title="Prediksi Risiko Diabetes", layout="wide")
st.title("Aplikasi Prediksi Risiko Diabetes")
st.write(
    "Aplikasi ini menggunakan model Machine Learning untuk memprediksi risiko diabetes pada pasien. "
    "Silakan masukkan data medis pasien pada panel di sebelah kiri untuk melihat hasil prediksi."
)

# Menampilkan metrik performa model jika berhasil dimuat
if metrics is not None:
    st.subheader("Performa Model yang Digunakan")
    st.info(
        f"Model terbaik ({metrics['model']} dengan seleksi fitur {metrics['feature_set']}) dipilih berdasarkan F1-Score tertinggi."
    )
    
    # Menampilkan metrik dalam kolom yang rapi
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Akurasi", f"{metrics['accuracy']:.2%}")
    col2.metric("Presisi", f"{metrics['precision']:.2%}")
    col3.metric("Recall", f"{metrics['recall']:.2%}")
    col4.metric("F1-Score", f"{metrics['f1']:.4f}")
    col5.metric("AUC", f"{metrics.get('roc_auc', 0):.4f}") # Gunakan .get untuk kompatibilitas

# Sidebar untuk input data
with st.sidebar:
    st.header("Parameter Input Pasien")
    st.write("Masukkan nilai untuk setiap parameter di bawah ini.")
    
    pregnancies = st.number_input("Jumlah Kehamilan (Pregnancies)", min_value=0, max_value=20, value=3, step=1, help="Jumlah total kehamilan.")
    glucose = st.number_input("Kadar Glukosa Plasma (Glucose)", min_value=0, max_value=250, value=120, help="Konsentrasi glukosa plasma 2 jam setelah OGTT. Normal: < 140 mg/dL.")
    blood_pressure = st.number_input("Tekanan Darah Diastolik (BloodPressure)", min_value=0, max_value=130, value=70, help="Tekanan darah diastolik (mm Hg). Normal: < 80 mm Hg.")
    skin_thickness = st.number_input("Ketebalan Lipatan Kulit (SkinThickness)", min_value=0, max_value=100, value=20, help="Ketebalan lipatan kulit trisep (mm). Indikator lemak tubuh.")
    insulin = st.number_input("Kadar Insulin Serum (Insulin)", min_value=0, max_value=900, value=80, help="Kadar insulin serum 2 jam setelah OGTT (mu U/ml). Normal: < 100 mu U/ml.")
    bmi = st.number_input("Indeks Massa Tubuh (BMI)", min_value=0.0, max_value=70.0, value=32.0, format="%.1f", help="BMI (kg/m²). Normal: 18.5-24.9. Obesitas: > 30.")
    dpf = st.number_input("Fungsi Silsilah Diabetes (DPF)", min_value=0.0, max_value=2.5, value=0.47, format="%.3f", help="Skor risiko diabetes berdasarkan riwayat keluarga.")
    age = st.number_input("Usia (Age)", min_value=18, max_value=100, value=33, help="Usia pasien dalam tahun.")

# Tombol prediksi utama
predict_button = st.button("Prediksi Risiko", type="primary", use_container_width=True)

# Proses prediksi
if all(v is not None for v in [model, scaler, explainer, metrics]) and predict_button:
    st.header("Hasil Analisis Risiko Diabetes")

    feature_names_original = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]], columns=feature_names_original)

    input_data_scaled = scaler.transform(input_data)
    
    rfe_feature_names = ['Glucose', 'BloodPressure', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    rfe_feature_indices = [feature_names_original.index(f) for f in rfe_feature_names]
    final_input = input_data_scaled[:, rfe_feature_indices]

    try:
        prediction = model.predict(final_input)
        prediction_proba = model.predict_proba(final_input)

        col1, col2 = st.columns(2)
        with col1:
            if prediction[0] == 1:
                st.warning("**Status: Pasien Berisiko Tinggi Terkena Diabetes**")
            else:
                st.success("**Status: Pasien Berisiko Rendah Terkena Diabetes**")
        
        with col2:
            st.metric(label="Probabilitas Risiko Diabetes", value=f"{prediction_proba[0][1]*100:.2f}%")

        st.subheader("Faktor-Faktor yang Mempengaruhi Prediksi (Analisis SHAP)")

        shap_df = pd.DataFrame(final_input, columns=rfe_feature_names)
        shap_values = explainer.shap_values(shap_df)
        
        st.write("Plot di bawah ini menunjukkan bagaimana setiap fitur 'mendorong' prediksi dari nilai dasar (rata-rata prediksi model) ke hasil akhir. Fitur berwarna merah meningkatkan risiko, sedangkan yang biru menurunkannya.")
        
        st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1], shap_df, matplotlib=False), height=150)

        shap_values_instance = shap_values[1][0]
        feature_impact = sorted(zip(rfe_feature_names, shap_values_instance), key=lambda x: abs(x[1]), reverse=True)
        
        st.write("#### Ringkasan Faktor Kunci:")
        for feature, impact in feature_impact[:3]:
            if impact > 0:
                st.markdown(f"- **{feature}**: Nilai yang dimasukkan secara signifikan **meningkatkan** risiko.")
            else:
                st.markdown(f"- **{feature}**: Nilai yang dimasukkan secara signifikan **menurunkan** risiko.")
            
    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan analisis: {e}")

# Disclaimer
st.markdown("---")
st.warning(
    "**Disclaimer:** Aplikasi ini adalah alat bantu dan tidak menggantikan konsultasi medis profesional. "
    "Hasil prediksi tidak boleh digunakan sebagai diagnosis tunggal. Selalu konsultasikan dengan dokter."
)
