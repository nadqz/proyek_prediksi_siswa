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

# Path file (Asumsi semua model 5-fitur ada dan bekerja)
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

# Definisikan OPSI KATEGORI untuk UI (Hanya untuk tampilan, tidak digunakan di model)
CATEGORICAL_OPTIONS = {
    'gender': ['Female', 'Male'],
    'part_time_job': ['No', 'Yes'],
    'diet_quality': ['Good', 'Fair', 'Poor', 'Average'],
    'parental_education_level': ['Master', 'Bachelor', 'High School', 'Some College', 'PhD'],
    'internet_quality': ['Good', 'Average', 'Poor'],
    'extracurricular_participation': ['Yes', 'No']
}

DL_3D_STANDARD = ["LSTM"] 
DL_3D_CNN = ["CNN"] 

# ... (Fungsi load_all_assets() tetap sama) ...
@st.cache_resource
def load_all_assets():
    models = {}
    # ... (Loading logic for scaler, ML, and DL models) ...
    try:
        scaler = joblib.load(SCALER_PATH)
    except FileNotFoundError:
        st.error(f"Error: Scaler file not found at {SCALER_PATH}. Check folder structure.")
        st.stop()
    
    # ... (Loading ML and DL models) ...
    all_paths = {**ML_PATHS, **DL_PATHS}
    for name, path in all_paths.items():
        try:
            if name in DL_PATHS:
                models[name] = load_model(path)
            else:
                models[name] = joblib.load(path)
        except FileNotFoundError:
            st.warning(f"Model {name} not found at {path}. Skipping.")
        except Exception as e:
            st.warning(f"Error loading {name}: {e}. Skipping.")

    return models, scaler

MODELS, SCALER = load_all_assets()

# ==============================================================================
# 2. FUNGSI PREPROCESSING DAN PREDIKSI (Disempurnakan)
# ==============================================================================

def preprocess_input(input_data, scaler):
    """
    Mengambil input dari form, memfilter HANYA 5 FITUR, 
    dan mengembalikan DataFrame (untuk ML) dan Array (untuk DL).
    """
    
    # 1. Buat DataFrame UTUH dari input form
    full_input_df = pd.DataFrame([input_data])
    
    # 2. FILTER: Ambil hanya 5 fitur numerik yang dibutuhkan model
    input_df_filtered = full_input_df[FEATURES_USED]
    
    # 3. Scaling (Untuk DL dan LR)
    X_scaled = scaler.transform(input_df_filtered)
    
    return input_df_filtered, X_scaled # Mengembalikan DataFrame dan Array

# Fungsi predict_score (menggunakan logika fix terakhir)
def predict_score(model_name, model, input_df, X_scaled):
    
    if model_name in DL_3D_STANDARD: 
        # LSTM: (samples, timesteps=1, features=5)
        X_final = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    elif model_name in DL_3D_CNN:
        # CNN: (samples, sequence_length=5, feature_depth=1)
        X_final = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
    else: 
        # DNN, RF, DT, LR: Gunakan DataFrame (untuk robustness ML) atau Array
        if model_name in ML_PATHS:
             X_final = input_df # Gunakan DataFrame untuk robustness ML
        else: # DNN
             X_final = X_scaled # Gunakan Array 2D untuk DNN

    try:
        prediction = model.predict(X_final) 
    except Exception as e:
        st.error(f"Prediction Failed for {model_name}.")
        st.error(f"Penyebab: {type(e).__name__} - {e}")
        st.info("Kemungkinan masalah versi scikit-learn atau input data.")
        st.stop()
    
    if model_name in DL_PATHS:
        return float(prediction[0][0]) 
    else:
        return float(prediction[0])

# ==============================================================================
# 3. STRUKTUR ANTARMUKA STREAMLIT (FORMULIR LENGKAP)
# ==============================================================================

st.set_page_config(page_title="Formulir Prediksi Nilai Siswa", layout="wide")
st.title("🎯 Formulir Prediksi Nilai Ujian Siswa")
st.warning("⚠️ Perhatian: Model yang tersedia saat ini HANYA menggunakan 5 Fitur Numerik.")
st.info("Input di kolom 'Lingkungan & Gaya Hidup' tidak memengaruhi prediksi Anda.")
st.markdown("---")

# Input Model di Sidebar
st.sidebar.header("Pilih Model")
model_options = list(MODELS.keys())
selected_model_name = st.sidebar.selectbox(
    "Algoritma yang Digunakan:", 
    model_options,
    index=model_options.index("Random Forest") if "Random Forest" in model_options else 0
)

# --- BAGIAN INPUT FORM ---
with st.container():
    st.header("1. Data Dasar & Kebiasaan")
    colA, colB, colC = st.columns(3)
    
    with colA:
        # Kategori
        gender = st.selectbox("Jenis Kelamin", CATEGORICAL_OPTIONS['gender'])
        part_time_job = st.selectbox("Kerja Paruh Waktu", CATEGORICAL_OPTIONS['part_time_job'])
        
        # Numerik
        study_hours = st.slider("Jam Belajar per Hari", 0.0, 8.0, 4.0, 0.1)
        
    with colB:
        # Numerik
        attendance = st.slider("Persentase Kehadiran (%)", 60.0, 100.0, 90.0, 0.1)
        mental_health = st.slider("Rating Kesehatan Mental (1-10)", 1, 10, 7)
        
        # Kategori
        parental_education_level = st.selectbox("Edukasi Orang Tua", CATEGORICAL_OPTIONS['parental_education_level'])
        
    with colC:
        # Numerik
        sleep_hours = st.slider("Jam Tidur per Hari", 4.0, 10.0, 7.5, 0.1)
        exercise_freq = st.slider("Frekuensi Olahraga per Minggu", 0, 7, 3)
        
        # Kategori
        extracurricular_participation = st.selectbox("Ikut Ekstrakurikuler", CATEGORICAL_OPTIONS['extracurricular_participation'])
        

st.markdown("---")


# Kolom Tambahan yang DIABAIKAN (Hanya untuk keperluan Form)
with st.container():
    st.header("2. Lingkungan & Gaya Hidup (Tidak Dipakai Model)")
    colD, colE = st.columns(2)
    
    with colD:
        # Kategori
        diet_quality = st.selectbox("Kualitas Diet", CATEGORICAL_OPTIONS['diet_quality'])
        internet_quality = st.selectbox("Kualitas Internet", CATEGORICAL_OPTIONS['internet_quality'])
        
    with colE:
        # Numerik (Tambahan yang Tidak Digunakan Model Lama)
        age = st.slider("Usia (tahun) [Diabaikan]", 18, 25, 20)
        social_media_hours = st.slider("Jam Medsos per Hari [Diabaikan]", 0.0, 6.0, 2.0, 0.1)


st.markdown("---")

# Kumpulkan data input LENGKAP dari FORM
input_data = {
    "study_hours_per_day": study_hours,
    "attendance_percentage": attendance,
    "mental_health_rating": mental_health,
    "sleep_hours": sleep_hours,
    "exercise_frequency": exercise_freq,
    
    # Data tambahan yang diabaikan (Tapi dikumpulkan untuk input_df utuh)
    'gender': gender, 
    'part_time_job': part_time_job,
    'diet_quality': diet_quality,
    'parental_education_level': parental_education_level,
    'internet_quality': internet_quality,
    'extracurricular_participation': extracurricular_participation,
    'age': age,
    'social_media_hours': social_media_hours,
    'netflix_hours': st.slider("Jam Netflix per Hari [Diabaikan]", 0.0, 4.0, 1.0, 0.1, key='netflix_input')
}


# 4. Tombol Prediksi dan Output
st.markdown("---")

if st.button(f"Prediksi Nilai dengan {selected_model_name}"):
    
    if selected_model_name not in MODELS:
        st.error("Model yang dipilih tidak dapat dimuat. Cek file Anda.")
    else:
        # Panggil fungsi preprocessing yang BARU
        input_df, X_scaled = preprocess_input(input_data, SCALER)
        
        # Panggil fungsi prediksi yang BARU
        final_prediction = predict_score(selected_model_name, MODELS[selected_model_name], input_df, X_scaled)
        
        # Tampilkan Hasil
        st.success("✅ Prediksi Selesai!")
        st.markdown(f"## Nilai Ujian Prediksi: **{final_prediction:.2f}** / 100")
        st.info(f"Menggunakan Model: **{selected_model_name}**")
        
        if final_prediction >= 80:
            st.balloons()
