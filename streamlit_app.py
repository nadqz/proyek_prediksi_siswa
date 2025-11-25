import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# ==============================================================================
# 1. SETUP ASET MODEL DAN DATA
# ==============================================================================

# Definisi Fitur (Hanya 5 fitur numerik yang digunakan saat training)
FEATURES = [
    "study_hours_per_day", "attendance_percentage", "mental_health_rating", 
    "sleep_hours", "exercise_frequency"
]

# Path file
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

# Fungsi untuk memuat semua aset (Hanya dijalankan sekali)
@st.cache_resource
def load_all_assets():
    models = {}
    
    # Load Preprocessor
    try:
        scaler = joblib.load(SCALER_PATH)
    except FileNotFoundError:
        st.error(f"Error: Scaler file not found at {SCALER_PATH}. Check folder structure.")
        st.stop()
    
    # Load ML Models
    for name, path in ML_PATHS.items():
        try:
            models[name] = joblib.load(path)
        except FileNotFoundError:
            st.warning(f"ML Model {name} not found at {path}. Skipping.")
            
    # Load DL Models
    for name, path in DL_PATHS.items():
        try:
            models[name] = load_model(path)
        except FileNotFoundError:
            st.warning(f"DL Model {name} not found at {path}. Skipping.")

    return models, scaler

# Memuat semua aset
MODELS, SCALER = load_all_assets()


# ==============================================================================
# 2. FUNGSI PREPROCESSING DAN PREDIKSI
# ==============================================================================

def preprocess_input(input_data, scaler):
    """Mengubah input user (2D) menjadi format yang siap diprediksi (Scaled dan Reshaped)."""
    
    # 1. Ubah ke DataFrame
    input_df = pd.DataFrame([input_data], columns=FEATURES)
    
    # 2. Scaling (Wajib untuk semua model)
    X_scaled = scaler.transform(input_df)
    
    # 3. Reshape untuk DL (akan disesuaikan di fungsi prediksi)
    return X_scaled

# Tambahkan path model lain di sini jika diperlukan, tapi fokus pada DNN
DL_MODELS_3D_STANDARD = ["LSTM", "DNN"] # <--- HANYA LSTM yang butuh 3D (1, 1, F)

def predict_score(model_name, model, X_scaled):
    """Melakukan prediksi, menyesuaikan reshape jika model adalah DL (LSTM/CNN/DNN)."""

    # Cek apakah model membutuhkan input 3D (Sequential)
    if model_name in DL_3D_MODELS:
        # Reshape ke format 3D: (samples, timesteps, features)
        # Jika Anda menggunakan 14 fitur, bentuknya adalah: (1, 1, 14)
        X_final = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    
    # 1. Tentukan bentuk akhir X_final
    if model_name == "LSTM": 
        # Untuk LSTM (tabular), bentuk: (samples, timesteps=1, features=5)
        X_final = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
        
    elif model_name == "CNN":
        # Untuk CNN (tabular), bentuk: (samples, sequence_length=5, feature_depth=1)
        X_final = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
        
    else: # Model ML (RF, DT, LR)
        # Model ML hanya membutuhkan 2D
        X_final = X_scaled
    
    # Lakukan Prediksi
    prediction = model.predict(X_final)
    
    # Ambil nilai prediksi pertama dan konversi ke float (dari array)
    if model_name in DL_PATHS:
        return float(prediction[0][0]) 
    else:
        return float(prediction[0])


# ==============================================================================
# 3. STRUKTUR ANTARMUKA STREAMLIT
# ==============================================================================

st.set_page_config(
    page_title="Prediksi Nilai Siswa", 
    layout="wide",
    initial_sidebar_state="auto"
)

st.title("🎯 Prediksi Nilai Ujian Siswa (ML & DL Comparison)")
st.markdown("---")

# Input Model
st.sidebar.header("Pilih Model")
model_options = list(MODELS.keys())
selected_model_name = st.sidebar.selectbox(
    "Algoritma yang Digunakan:", 
    model_options,
    index=model_options.index("Random Forest") if "Random Forest" in model_options else 0
)

# Input Fitur (5 Fitur Numerik)
st.header("Input Data Kebiasaan Siswa")
col1, col2 = st.columns(2)

with col1:
    study_hours = st.slider("1. Jam Belajar per Hari", 
                            min_value=0.0, max_value=8.0, value=4.0, step=0.1)
    
    attendance = st.slider("2. Persentase Kehadiran (%)", 
                           min_value=50.0, max_value=100.0, value=85.0, step=0.1)
    
    sleep_hours = st.slider("3. Jam Tidur per Hari", 
                            min_value=4.0, max_value=10.0, value=7.0, step=0.1)

with col2:
    exercise_freq = st.slider("4. Frekuensi Olahraga per Minggu", 
                              min_value=0, max_value=7, value=3)
    
    mental_health = st.slider("5. Rating Kesehatan Mental (1-10)", 
                              min_value=1, max_value=10, value=6)

# Kumpulkan data input
input_data = {
    "study_hours_per_day": study_hours,
    "attendance_percentage": attendance,
    "mental_health_rating": mental_health,
    "sleep_hours": sleep_hours,
    "exercise_frequency": exercise_freq
}


# 4. Tombol Prediksi dan Output
st.markdown("---")

if st.button(f"Prediksi Nilai dengan {selected_model_name}"):
    
    if selected_model_name not in MODELS:
        st.error("Model yang dipilih tidak dapat dimuat. Cek file Anda.")
    else:
        # Preprocessing input
        X_scaled = preprocess_input(input_data, SCALER)
        
        # Prediksi
        final_prediction = predict_score(selected_model_name, MODELS[selected_model_name], X_scaled)
        
        # Tampilkan Hasil
        st.success("✅ Prediksi Selesai!")
        st.markdown(f"## Nilai Ujian Prediksi: **{final_prediction:.2f}** / 100")
        st.info(f"Menggunakan Model: **{selected_model_name}**")
        
        if final_prediction >= 80:
            st.balloons()
