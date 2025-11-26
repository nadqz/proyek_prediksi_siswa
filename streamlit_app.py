import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# ==============================================================================
# 1. SETUP ASET MODEL DAN DATA (Static Data Added)
# ==============================================================================

# Definisikan 5 Fitur yang BENAR-BENAR DIGUNAKAN OLEH MODEL
FEATURES_USED = [
    "study_hours_per_day", "attendance_percentage", "mental_health_rating", 
    "sleep_hours", "exercise_frequency"
]

# Path file dan Konfigurasi
SCALER_PATH = 'PREPROCESSOR/minmax_scaler.pkl'

# --- DATA STATIS ML (UNTUK BYPASS FILE .PKL YANG RUSAK) ---
# Asumsi: Anda mendapatkan nilai rata-rata dari prediksi test set Anda.
STATIC_ML_RESULTS = {
    "Random Forest": 78.50,  # Contoh Nilai Rata-rata Prediksi
    "Decision Tree": 74.10,
    "Linear Regression": 70.35
}
ML_PATHS = {
    # Kita tidak akan memuatnya, tetapi ini untuk referensi nama
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
    'gender': ['Perempuan', 'Laki-laki'],
    'part_time_job': ['Tidak', 'Ya'],
    'diet_quality': ['Baik', 'Cukup', 'Buruk', 'Rata-rata'],
    'parental_education_level': ['Master', 'Sarjana (Bachelor)', 'SMA/Sederajat', 'Kuliah Non-Gelar', 'Doktor (PhD)'],
    'internet_quality': ['Baik', 'Rata-rata', 'Buruk'],
    'extracurricular_participation': ['Ya', 'Tidak']
}


@st.cache_resource
def load_all_assets():
    """Fungsi hanya akan memuat Scaler dan Model DL."""
    models = {}
    
    try:
        scaler = joblib.load(SCALER_PATH)
    except FileNotFoundError:
        st.error(f"Error: Scaler file not found at {SCALER_PATH}.")
        st.stop()
    
    # HANYA MEMUAT MODEL DL
    for name, path in DL_PATHS.items():
        try:
            models[name] = load_model(path)
        except Exception as e:
            st.warning(f"Error loading DL Model {name}. Skipping.")

    # Tambahkan placeholder untuk model ML (agar logika perbandingan tidak error)
    for name in ML_PATHS.keys():
        models[name] = "STATIC_MODEL"
        
    return models, scaler

MODELS, SCALER = load_all_assets()


# ==============================================================================
# 2. FUNGSI PREPROCESSING DAN PREDIKSI (Disegmentasi)
# ==============================================================================

def preprocess_input(input_data, scaler):
    """Mengambil input form, memfilter 5 fitur utama, dan mengembalikan Array NumPy yang sudah di-scale."""
    full_input_df = pd.DataFrame([input_data])
    input_df_filtered = full_input_df[FEATURES_USED]
    X_scaled = scaler.transform(input_df_filtered)
    return X_scaled 


def predict_score(model_name, model, X_scaled):
    """
    Melakukan prediksi. 
    Jika model adalah ML, kembalikan nilai statis. Jika DL, lakukan prediksi live.
    """

    # --- LOGIKA STATIC (ML) ---
    if model == "STATIC_MODEL":
        # Kembalikan nilai statis sebagai placeholder yang aman
        return STATIC_ML_RESULTS.get(model_name, 70.00) 

    # --- LOGIKA LIVE (DL) ---
    # 1. Tentukan input akhir X_final
    if model_name in DL_3D_STANDARD: 
        X_final = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    elif model_name in DL_3D_CNN:
        X_final = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
    else: # DNN (menggunakan Array NumPy 2D)
        X_final = X_scaled
    
    # 2. Lakukan Prediksi
    try:
        prediction = model.predict(X_final, verbose=0)
    except Exception as e:
        st.error(f"Prediction Failed for {model_name} (DL). Log: {type(e).__name__}")
        return None
    
    # 3. Ambil nilai prediksi
    return float(prediction[0][0]) 

# ==============================================================================
# 3. STRUKTUR ANTARMUKA STREAMLIT (FINAL INTERAKTIF)
# ==============================================================================

st.set_page_config(page_title="Prediksi Nilai Siswa", layout="wide")
st.title("🎯 Formulir Interaktif: Prediksi Nilai Ujian Siswa")
st.caption("Jawablah pertanyaan di bawah ini untuk melihat estimasi nilai ujian Anda berdasarkan 6 model Machine Learning dan Deep Learning.")

# --- Bagian Peringatan ---
st.warning("⚠️ Perhatian: Model ML (Random Forest, Decision Tree, Linear Regression) menampilkan hasil STATIS karena file model tidak kompatibel dengan lingkungan deployment.")

# ... (Sisa UI form interaktif tetap sama, pastikan semua variabel terdaftar di input_data) ...

# 4. Tombol Prediksi dan Output
st.markdown("## 📊 Hasil Prediksi dan Perbandingan")

if st.button("Hitung Prediksi Semua Model", type="primary"):
    
    if not MODELS:
        st.error("Tidak ada model yang berhasil dimuat. Cek log error deployment.")
    else:
        # Preprocessing input
        X_scaled = preprocess_input(input_data, SCALER)
        results = []
        
        # LOOP MELALUI SEMUA MODEL
        for name, model in MODELS.items():
            prediction = predict_score(name, model, X_scaled)
            
            if prediction is not None:
                results.append({
                    "Algoritma": name,
                    "Tipe": "DL" if name in DL_PATHS else "ML",
                    "Prediksi Nilai": f"{prediction:.2f}"
                })

        # Tampilkan Hasil Perbandingan
        if results:
            results_df = pd.DataFrame(results)
            results_df['Prediksi Nilai'] = results_df['Prediksi Nilai'].astype(float)
            
            st.success("✅ Prediksi Selesai! Berikut Perbandingan Hasilnya:")
            
            # Cari model terbaik (nilai tertinggi)
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
