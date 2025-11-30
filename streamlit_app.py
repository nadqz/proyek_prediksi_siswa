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
# 2. FUNGSI PREPROCESSING DAN PREDIKSI (Koreksi Total)
# ==============================================================================

# Definisikan model berdasarkan kebutuhan input SHAPE
# Asumsi: DNN Anda tidak memiliki lapisan Flatten dan membutuhkan input 2D.
DL_3D_STANDARD = ["LSTM"]  # Model Sequential dengan timesteps=1
DL_3D_CNN = ["CNN"]        # Model Conv1D yang butuh (samples, features, 1)
DL_2D_MODELS = ["DNN"]     # Model Dense yang butuh input 2D (seperti ML)


def preprocess_input(input_data, scaler):
    """Mengubah input user menjadi DataFrame (belum di-scale)."""
    
    # 1. Ubah ke DataFrame (Ini yang akan digunakan ML)
    input_df = pd.DataFrame([input_data], columns=FEATURES)
    
    # 2. Scaling (Ini yang akan digunakan DL)
    X_scaled = scaler.transform(input_df)
    
    # Mengembalikan keduanya untuk fleksibilitas: 
    # DataFrame untuk ML, Array ter-scale untuk DL.
    return input_df, X_scaled

# Ganti panggilan fungsi di bawah ini:
# final_prediction = predict_score(selected_model_name, MODELS[selected_model_name], X_scaled)

# Ganti menjadi:
# final_prediction = predict_score(selected_model_name, MODELS[selected_model_name], input_df, X_scaled)
# (Koreksi ini akan dilakukan di bagian 4 kode Streamlit)

def predict_score(model_name, model, input_df, X_scaled):
    """
    Melakukan prediksi dengan menyesuaikan input: 
    - DL: X_scaled (array) dengan reshape 3D
    - ML/DNN: input_df (DataFrame) atau X_scaled (array 2D)
    """

    # 1. Tentukan input akhir X_final
    if model_name in DL_PATHS:

        # Logika 3D untuk DL (sudah fix untuk LSTM/CNN/DNN)
        if model_name == "LSTM": 
            X_final = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
        elif model_name == "CNN":
            X_final = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
        else: # DNN (menggunakan array 2D jika model dilatih tanpa Flatten, atau 3D jika ada Flatten)
              # Karena model Anda adalah DNN lama, kita biarkan 2D saja
            X_final = X_scaled
        
    else: 
        # Model ML (RF, DT, LR) - GUNAKAN DATAFRAME AGAR LEBIH ROBUST
        # Ini mengatasi error input/versi di scikit-learn
        X_final = input_df 
    
    # 2. Lakukan Prediksi dengan Error Catching yang Informatif
    try:
        # Hapus verbose=0 dari ML models (prediction = model.predict(X_final))
        prediction = model.predict(X_final) 
    except Exception as e:
        # Error Catching yang jelas
        st.error(f"Prediction Failed for {model_name}.")
        st.error(f"Penyebab: {type(e).__name__} - {e}")
        st.info("Kemungkinan besar model ini dilatih dengan versi scikit-learn yang berbeda.")
        st.stop()
    
    # 3. Ambil nilai prediksi
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
        # Panggil fungsi preprocessing yang baru
        input_df, X_scaled = preprocess_input(input_data, SCALER)
        
        # Panggil fungsi prediksi yang baru
        final_prediction = predict_score(selected_model_name, MODELS[selected_model_name], input_df, X_scaled)
        
        # Tampilkan Hasil
        st.success("✅ Prediksi Selesai!")
        st.markdown(f"## Nilai Ujian Prediksi: **{final_prediction:.2f}** / 100")
        st.info(f"Menggunakan Model: **{selected_model_name}**")
        
        if final_prediction >= 80:
            st.balloons()
