import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# ==============================================================================
# 1. SETUP ASET MODEL DAN DATA
# ==============================================================================

# Definisikan 5 Fitur yang BENAR-BENAR DIGUNAKAN OLEH MODEL
FEATURES_USED = [
    "study_hours_per_day", "attendance_percentage", "mental_health_rating", 
    "sleep_hours", "exercise_frequency"
]

# Path files dan Konfigurasi
SCALER_PATH = 'PREPROCESSOR/minmax_scaler.pkl'
ML_PATHS = {
    "Random Forest": 'ML_MODELS/random_forest_model.pkl',
    "Decision Tree": 'ML_MODELS/decision_tree_model.pkl',
    "Linear Regression": 'ML_MODELS/linear_regression_model.pkl'
}
DL_PATHS = {
    "LSTM": 'DL_MODELS/lstm_model.keras',
    "CNN": 'DL_MODELS/cnn_model.keras',
    "DNN": 'DL_MODELS/dnn_model.keras'
}

# Definisikan model berdasarkan kebutuhan input SHAPE
DL_3D_STANDARD = ["LSTM"] 
DL_3D_CNN = ["CNN"] 

# Definisikan OPSI KATEGORI untuk UI
CATEGORICAL_OPTIONS = {
    'gender': ['Female', 'Male'],
    'part_time_job': ['Tidak', 'Ya'],
    'diet_quality': ['Baik', 'Cukup', 'Buruk', 'Rata-rata'],
    'parental_education_level': ['Master', 'Sarjana (Bachelor)', 'SMA/Sederajat', 'Kuliah Non-Gelar', 'Doktor (PhD)'],
    'internet_quality': ['Baik', 'Rata-rata', 'Buruk'],
    'extracurricular_participation': ['Ya', 'Tidak']
}

@st.cache_resource
def load_all_assets():
    models = {}
    
    try:
        scaler = joblib.load(SCALER_PATH)
    except FileNotFoundError:
        st.error(f"Error: Scaler file not found at {SCALER_PATH}. Check folder structure.")
        st.stop()
    
    all_paths = {**ML_PATHS, **DL_PATHS}
    for name, path in all_paths.items():
        try:
            if name in DL_PATHS:
                models[name] = load_model(path)
            else:
                models[name] = joblib.load(path)
        except Exception as e:
            st.warning(f"Error loading {name}: {e}. Skipping.")

    return models, scaler

MODELS, SCALER = load_all_assets()

# ==============================================================================
# 2. FUNGSI PREPROCESSING DAN PREDIKSI (FINAL FIX)
# ==============================================================================

def preprocess_input(input_data, scaler):
    """Mengambil input form, memfilter 5 fitur, dan mengembalikan DataFrame dan Array."""
    
    # 1. Buat DataFrame UTUH dari input form
    full_input_df = pd.DataFrame([input_data])
    
    # 2. FILTER: Ambil hanya 5 fitur numerik yang dibutuhkan model
    input_df_filtered = full_input_df[FEATURES_USED]
    
    # 3. Scaling (Untuk DL dan LR)
    X_scaled = scaler.transform(input_df_filtered)
    
    return input_df_filtered, X_scaled # Mengembalikan DataFrame dan Array

def predict_score(model_name, model, input_df, X_scaled):
    """Melakukan prediksi, menyesuaikan input format (DataFrame/Array) dan shape."""

    # 1. Tentukan input akhir X_final
    if model_name in DL_3D_STANDARD: 
        # LSTM: (samples, timesteps=1, features=5)
        X_final = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    elif model_name in DL_3D_CNN:
        # CNN: (samples, sequence_length=5, feature_depth=1)
        X_final = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
    elif model_name in ["DNN"] or model_name in ML_PATHS:
        # DNN, RF, DT, LR: Menggunakan DataFrame (ML) atau Array 2D (DNN)
        if model_name in ML_PATHS:
             X_final = input_df  # DataFrame untuk ML Robustness
        else:
             X_final = X_scaled  # Array 2D untuk DNN
    else: 
        # Fallback (seharusnya tidak tercapai)
        X_final = input_df
    
    # 2. Lakukan Prediksi dengan Error Catching yang Informatif
    try:
        prediction = model.predict(X_final, verbose=0)
    except Exception as e:
        st.error(f"Prediction Failed for {model_name}.")
        st.error(f"Penyebab: {type(e).__name__} - {e}")
        st.stop()
    
    # 3. Ambil nilai prediksi
    if model_name in DL_PATHS:
        return float(prediction[0][0]) 
    else:
        return float(prediction[0])

# ==============================================================================
# 3. STRUKTUR ANTARMUKA STREAMLIT (FINAL INTERAKTIF)
# ==============================================================================

st.set_page_config(page_title="Prediksi Nilai Siswa", layout="wide", initial_sidebar_state="auto")
st.title("💡 Form Interaktif: Prediksi Nilai Ujian Siswa")

# --- Bagian Input Form tetap sama dengan versi interaktif terakhir ---

st.sidebar.header("⚙️ Konfigurasi Model")
# ... (Logika sidebar tetap sama) ...
selected_model_name = st.sidebar.selectbox(
    "Algoritma yang Digunakan:", 
    list(MODELS.keys()),
    index=list(MODELS.keys()).index("Random Forest") if "Random Forest" in MODELS else 0
)

st.warning("⚠️ Perhatian: Model yang tersedia saat ini HANYA menggunakan 5 Fitur Numerik.")
st.info("Input di Bagian 2 (Data Pendukung) tidak memengaruhi prediksi Anda.")

# ... (Semua UI form interaktif di bagian 1 dan 2 tetap sama) ...

# Kumpulkan data input LENGKAP dari FORM (Menggunakan nilai default untuk variabel non-input)
# ... (Pastikan semua widget dari UI terdaftar di sini) ...
# ... (Contoh: menggunakan nilai default untuk memastikan input_data lengkap) ...

# Input Widgets
col_main_1, col_main_2, col_main_3 = st.columns(3)
with col_main_1:
    study_hours = st.number_input("Jam Belajar per Hari", min_value=0.0, max_value=8.0, value=4.0, step=0.5, key='study')
    gender = st.radio("Jenis Kelamin", CATEGORICAL_OPTIONS['gender'], horizontal=True, key='gender_input')
    part_time_job = st.radio("Kerja Paruh Waktu", CATEGORICAL_OPTIONS['part_time_job'], horizontal=True, key='job_input')
with col_main_2:
    attendance = st.number_input("Persentase Kehadiran (%)", min_value=60.0, max_value=100.0, value=90.0, step=0.1, key='attn')
    mental_health = st.radio("Rating Kesehatan Mental (1-10)", options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], index=6, horizontal=True, key='mental')
    parental_education_level = st.selectbox("Edukasi Orang Tua", CATEGORICAL_OPTIONS['parental_education_level'], key='parental')
with col_main_3:
    sleep_hours = st.slider("Jam Tidur per Hari", min_value=4.0, max_value=10.0, value=7.0, step=0.5, key='sleep')
    exercise_freq = st.selectbox("Frekuensi Olahraga per Minggu", options=list(range(0, 8)), index=3, key='exercise')
    extracurricular_participation = st.radio("Ikut Ekstrakurikuler", CATEGORICAL_OPTIONS['extracurricular_participation'], horizontal=True, key='extra_input')

# Tambahan (Diabaikan)
col_add_1, col_add_2 = st.columns(2)
with col_add_1:
    diet_quality = st.selectbox("Kualitas Diet", CATEGORICAL_OPTIONS['diet_quality'], key='diet')
    internet_quality = st.selectbox("Kualitas Internet [Diabaikan]", CATEGORICAL_OPTIONS['internet_quality'], key='internet')
with col_add_2:
    age = st.number_input("Usia (Tahun) [Diabaikan]", min_value=16, max_value=30, value=20, key='age_input')
    social_media_hours = st.number_input("Jam Medsos per Hari [Diabaikan]", 0.0, 6.0, 2.0, step=0.5, key='socmed_input')
    netflix_hours = st.slider("Jam Netflix per Hari [Diabaikan]", 0.0, 4.0, 1.0, 0.1, key='netflix_input')


# Kumpulkan data input LENGKAP dari FORM
input_data = {
    "study_hours_per_day": study_hours,
    "attendance_percentage": attendance,
    "mental_health_rating": mental_health,
    "sleep_hours": sleep_hours,
    "exercise_frequency": exercise_freq,
    
    # Fitur Pendukung (Diabaikan oleh model, tapi dikumpulkan)
    'gender': gender, 'part_time_job': part_time_job, 'diet_quality': diet_quality,
    'parental_education_level': parental_education_level, 'internet_quality': internet_quality,
    'extracurricular_participation': extracurricular_participation,
    'age': age, 'social_media_hours': social_media_hours, 'netflix_hours': netflix_hours
}


# 4. Tombol Prediksi dan Output (Tetap Sama)
st.markdown("## 📊 Hasil Prediksi dan Perbandingan")

if st.button("Hitung Prediksi Semua Model", type="primary"):
    
    if not MODELS:
        st.error("Tidak ada model yang berhasil dimuat. Cek path file di terminal Anda.")
    else:
        input_df, X_scaled = preprocess_input(input_data, SCALER)
        results = []
        
        for name, model in MODELS.items():
            prediction = predict_score(name, model, input_df, X_scaled)
            
            if prediction is not None:
                results.append({
                    "Algoritma": name,
                    "Tipe": "DL" if name in DL_PATHS else "ML",
                    "Prediksi Nilai": f"{prediction:.2f}"
                })

        if results:
            results_df = pd.DataFrame(results)
            results_df['Prediksi Nilai'] = results_df['Prediksi Nilai'].astype(float)
            
            st.success("✅ Prediksi Selesai! Berikut Perbandingan Hasilnya:")
            
            best_model_name = results_df.sort_values(by="Prediksi Nilai", ascending=False).iloc[0]["Algoritma"]
            best_score = results_df["Prediksi Nilai"].max()

            col_res_1, col_res_2 = st.columns([1, 2])
            
            with col_res_1:
                st.metric(
                    label="Model Terbaik", 
                    value=best_model_name, 
                    delta=f"{best_score:.2f} Poin"
                )
            
            with col_res_2:
                st.dataframe(
                    results_df.sort_values(by="Prediksi Nilai", ascending=False).set_index("Algoritma"),
                    use_container_width=True
                )
                
            if best_score >= 80:
                st.balloons()
