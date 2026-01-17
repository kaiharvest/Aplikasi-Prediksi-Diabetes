
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
        return None, None, None, None, None

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    explainer = shap.TreeExplainer(model)
    
    # Memuat hasil dan mencari metrik model terbaik
    results_df = pd.read_csv(results_path)
    best_model_metrics = results_df.loc[results_df['f1'].idxmax()]
    
    # Mendapatkan nama fitur yang digunakan oleh model terbaik
    feature_set_name = best_model_metrics['feature_set']
    
    # Daftar fitur asli
    feature_names_original = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    
    # Mendefinisikan set fitur (ini harus konsisten dengan `main.py`)
    # Ini adalah pendekatan yang disederhanakan. Idealnya, file `main.py` akan menyimpan indeks ini.
    selection_sets = {
        "All": list(range(len(feature_names_original))),
        "RFE": [1, 2, 5, 6, 7],
        "Boruta": [0, 1, 5, 6, 7],
        "GA": [1, 2, 4, 5, 6, 7],
        "PSO": [0, 1, 2, 4, 5, 7],
        "GWO": [0, 1, 2, 4, 5, 7],
        "TopSHAP5": [1, 0, 5, 7, 6] 
    }
    
    best_feature_indices = selection_sets.get(feature_set_name, list(range(len(feature_names_original))))
    best_feature_names = [feature_names_original[i] for i in best_feature_indices]

    return model, scaler, explainer, best_model_metrics, (best_feature_indices, best_feature_names)

# Memuat aset
model, scaler, explainer, metrics, best_features = load_assets()

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
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Akurasi", f"{metrics['accuracy']:.2%}")
    col2.metric("Presisi", f"{metrics['precision']:.2%}")
    col3.metric("Recall", f"{metrics['recall']:.2%}")
    col4.metric("F1-Score", f"{metrics['f1']:.4f}")
    col5.metric("AUC", f"{metrics.get('roc_auc', 0.0):.4f}")

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
if all(v is not None for v in [model, scaler, explainer, metrics, best_features]) and predict_button:
    st.header("Hasil Analisis Risiko Diabetes")

    feature_names_original = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]], columns=feature_names_original)

    input_data_scaled = scaler.transform(input_data)
    
    best_feature_indices, best_feature_names = best_features
    final_input = input_data_scaled[:, best_feature_indices]

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

        shap_df = pd.DataFrame(final_input, columns=best_feature_names)
        shap_values = explainer.shap_values(shap_df)

        st.write("Visualisasi di bawah ini menunjukkan seberapa besar pengaruh masing-masing parameter klinis terhadap prediksi risiko diabetes pasien. Parameter berwarna merah menambah risiko, sedangkan yang biru mengurangi risiko.")

        # Logika robust untuk menangani berbagai struktur output dari SHAP explainer.
        base_value = explainer.expected_value
        shap_values_raw = shap_values

        # Tangani berbagai format output SHAP
        # Berdasarkan tes sebelumnya, untuk model klasifikasi biner,
        # SHAP bisa mengembalikan array 3D dengan bentuk (n_samples, n_features, n_classes)
        if isinstance(shap_values_raw, list):
            # Jika output berupa list (multi-class), ambil untuk kelas positif (indeks 1)
            shap_values_for_class = shap_values_raw[1] if len(shap_values_raw) > 1 else shap_values_raw[0]
        else:
            # Jika output bukan list, cek bentuk array
            if len(shap_values_raw.shape) == 3:
                # Array 3D: (n_samples, n_features, n_classes)
                # Ambil semua fitur untuk kelas positif (indeks 1) dari semua sampel
                # Lalu ambil hanya instance pertama
                shap_values_for_class = shap_values_raw[:, :, 1]  # Ambil kelas positif
            elif len(shap_values_raw.shape) == 2 and shap_values_raw.shape[1] == 2:
                # Array 2D dengan 2 kolom (satu untuk tiap kelas), ambil kelas positif
                shap_values_for_class = shap_values_raw[:, 1]

        # Sekarang ambil hanya instance pertama untuk analisis
        if len(shap_values_for_class.shape) == 2:
            # Jika masih 2D, ambil instance pertama
            shap_values_instance = shap_values_for_class[0, :]  # Ambil instance pertama
        else:
            # Jika sudah 1D, gunakan langsung
            shap_values_instance = shap_values_for_class

        # Tangani base_value untuk multi-class
        if isinstance(base_value, (list, np.ndarray)) and len(base_value) > 1:
            # Ambil base value untuk kelas positif (indeks 1)
            base_value_final = base_value[1] if len(base_value) > 1 else base_value[0]
        else:
            base_value_final = base_value

        if isinstance(base_value_final, np.ndarray):
            base_value_final = base_value_final.item()

        # Tampilkan force plot
        try:
            st_shap(shap.force_plot(base_value_final, shap_values_instance, shap_df.iloc[0], matplotlib=False), height=150)
        except Exception as plot_error:
            st.warning(f"Gambar visualisasi tidak dapat ditampilkan: {plot_error}")

        # Urutkan berdasarkan nilai absolut (dampak terbesar terhadap prediksi)
        feature_impact = sorted(zip(best_feature_names, shap_values_instance), key=lambda x: abs(x[1]), reverse=True)

        # Tampilkan analisis faktor-faktor penting
        st.write("#### Interpretasi Klinis Faktor-Faktor Utama:")

        # Kelompokkan fitur berdasarkan dampaknya
        increasing_factors = [(f, v) for f, v in feature_impact if v > 0]
        decreasing_factors = [(f, v) for f, v in feature_impact if v < 0]

        # Tampilkan faktor yang meningkatkan risiko
        if increasing_factors:
            st.write("**Parameter yang Meningkatkan Risiko Diabetes:**")
            for feature, impact in increasing_factors[:3]:  # Ambil 3 faktor teratas
                # Ambil nilai input asli untuk konteks
                original_value = input_data[feature].iloc[0]

                # Penjelasan berdasarkan nilai SHAP dan konteks medis
                if feature == "Glucose":
                    if original_value > 140:
                        explanation = f"Kadar glukosa darah pasien ({original_value} mg/dL) sangat tinggi, jauh di atas ambang batas normal (< 140 mg/dL setelah uji toleransi glukosa oral), yang merupakan indikator kuat diabetes mellitus."
                    elif original_value > 120:
                        explanation = f"Kadar glukosa darah pasien ({original_value} mg/dL) berada dalam kategori pra-diabetes, menunjukkan gangguan toleransi glukosa yang meningkatkan risiko berkembangnya diabetes."
                    else:
                        explanation = f"Meskipun kadar glukosa darah pasien ({original_value} mg/dL) dalam rentang normal, parameter ini tetap berkontribusi terhadap peningkatan risiko dalam konteks kombinasi faktor lain."
                elif feature == "BMI":
                    if original_value > 30:
                        explanation = f"Indeks Massa Tubuh (BMI) pasien ({original_value}) menunjukkan obesitas (BMI ≥ 30), yang merupakan faktor risiko utama diabetes tipe 2 karena meningkatkan resistensi insulin."
                    elif original_value > 25:
                        explanation = f"BMI pasien ({original_value}) menunjukkan overweight (25 ≤ BMI < 30), yang meningkatkan risiko diabetes tipe 2 meskipun tidak sebesar obesitas."
                    else:
                        explanation = f"Meskipun BMI pasien ({original_value}) dalam rentang normal, parameter ini tetap berkontribusi terhadap peningkatan risiko dalam konteks kombinasi faktor lain."
                elif feature == "Age":
                    if original_value > 45:
                        explanation = f"Usia pasien ({original_value} tahun) merupakan faktor risiko karena sensitivitas insulin cenderung menurun seiring bertambahnya usia, terutama setelah usia 45 tahun."
                    else:
                        explanation = f"Meskipun usia pasien ({original_value} tahun) bukan faktor risiko tinggi secara mandiri, parameter ini tetap berkontribusi dalam kombinasi dengan faktor lain."
                elif feature == "Pregnancies":
                    explanation = f"Jumlah kehamilan ({int(original_value)}) dapat meningkatkan risiko diabetes, terutama jika pasien memiliki riwayat diabetes gestasional di masa lalu."
                elif feature == "Insulin":
                    explanation = f"Kadar insulin ({original_value} mu U/ml) yang tinggi menunjukkan adanya resistensi insulin, kondisi awal yang umum sebelum berkembangnya diabetes tipe 2."
                elif feature == "DiabetesPedigreeFunction":
                    explanation = f"Nilai fungsi silsilah diabetes ({original_value}) menunjukkan adanya riwayat keturunan diabetes dalam keluarga, yang meningkatkan predisposisi genetik terhadap diabetes."
                elif feature == "BloodPressure":
                    explanation = f"Tekanan darah diastolik ({original_value} mm Hg) yang tinggi menunjukkan hipertensi, yang sering berkaitan dengan sindrom metabolik dan resistensi insulin."
                elif feature == "SkinThickness":
                    explanation = f"Ketebalan lipatan kulit triceps ({original_value} mm) yang tinggi menunjukkan persentase lemak tubuh yang lebih tinggi, yang berkaitan dengan resistensi insulin."
                else:
                    explanation = f"Nilai {feature} ({original_value}) berkontribusi pada peningkatan risiko diabetes dalam kombinasi dengan faktor-faktor lain."

                st.markdown(f"- **{feature}**: Meningkatkan risiko sebesar {abs(impact):.3f}")
                st.caption(explanation)

        # Tampilkan faktor yang menurunkan risiko
        if decreasing_factors:
            st.write("**Parameter yang Mengurangi Risiko Diabetes:**")
            for feature, impact in decreasing_factors[:3]:  # Ambil 3 faktor teratas
                original_value = input_data[feature].iloc[0]

                # Penjelasan berdasarkan nilai SHAP dan konteks medis
                if feature == "Glucose":
                    explanation = f"Kadar glukosa darah ({original_value} mg/dL) pasien dalam rentang normal, yang menunjukkan toleransi glukosa yang baik dan mengurangi risiko diabetes."
                elif feature == "BMI":
                    explanation = f"BMI ({original_value}) pasien dalam rentang sehat (18.5-24.9), yang menunjukkan berat badan ideal dan mengurangi risiko resistensi insulin."
                elif feature == "Age":
                    explanation = f"Usia muda ({original_value} tahun) merupakan faktor protektif karena sensitivitas insulin biasanya lebih baik pada usia muda."
                elif feature == "Pregnancies":
                    explanation = f"Jumlah kehamilan ({int(original_value)}) relatif rendah, yang mengurangi kemungkinan riwayat diabetes gestasional."
                elif feature == "Insulin":
                    explanation = f"Kadar insulin ({original_value} mu U/ml) pasien dalam rentang normal, menunjukkan sensitivitas insulin yang baik dan fungsi pankreas yang sehat."
                elif feature == "DiabetesPedigreeFunction":
                    explanation = f"Nilai fungsi silsilah diabetes ({original_value}) relatif rendah, menunjukkan riwayat keluarga yang minim terhadap diabetes."
                elif feature == "BloodPressure":
                    explanation = f"Tekanan darah diastolik ({original_value} mm Hg) pasien dalam rentang normal, yang menunjukkan tidak adanya hipertensi sebagai faktor risiko tambahan."
                elif feature == "SkinThickness":
                    explanation = f"Ketebalan lipatan kulit ({original_value} mm) yang rendah menunjukkan persentase lemak tubuh yang sehat, mengurangi risiko resistensi insulin."
                else:
                    explanation = f"Nilai {feature} ({original_value}) berkontribusi pada penurunan risiko diabetes dalam kombinasi dengan faktor lain."

                st.markdown(f"- **{feature}**: Mengurangi risiko sebesar {abs(impact):.3f}")
                st.caption(explanation)

        # Tampilkan ringkasan klinis
        st.write("#### Ringkasan Klinis:")
        st.write("Berikut adalah tiga parameter yang paling berpengaruh terhadap prediksi risiko diabetes pasien:")
        for feature, impact in feature_impact[:3]:
            original_value = input_data[feature].iloc[0]
            if impact > 0:
                st.markdown(f"- **{feature}** (Nilai: {original_value}): Meningkatkan risiko sebesar {abs(impact):.3f} satuan")
            else:
                st.markdown(f"- **{feature}** (Nilai: {original_value}): Mengurangi risiko sebesar {abs(impact):.3f} satuan")

        st.info("Catatan: Nilai SHAP menunjukkan seberapa besar masing-masing parameter menyimpang dari rata-rata populasi dalam mempengaruhi prediksi risiko diabetes.")
            
    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan analisis: {e}")

# Disclaimer
st.markdown("---")
st.warning(
    "**Disclaimer:** Aplikasi ini adalah alat bantu dan tidak menggantikan konsultasi medis profesional. "
    "Hasil prediksi tidak boleh digunakan sebagai diagnosis tunggal. Selalu konsultasikan dengan dokter."
)
