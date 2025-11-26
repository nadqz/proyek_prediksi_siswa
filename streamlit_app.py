import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# ==============================================================================
# 1. SETUP ASET MODEL DAN DATA (Dibiarkan SAMA)
# ==============================================================================

# Definisi Fitur (Hanya 5 fitur numerik yang digunakan saat training)
FEATURES_USED = [
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

# Definisikan OPSI KATEGORI untuk UI
CATEGORICAL_OPTIONS = {
    'gender': ['Female', 'Male'],
    'part_time_job': ['Tidak', 'Ya'],
    'diet_quality': ['Baik', 'Cukup', 'Buruk', 'Rata-rata'],
    'parental_education_level': ['Master', 'Sarjana (Bachelor)', 'SMA/Sederajat', 'Kuliah Non-Gelar', 'Doktor (PhD)'],
    'internet_quality': ['Baik', 'Rata-rata', 'Buruk'],
    'extracurricular_participation': ['Ya', 'Tidak']
}

DL_3D_STANDARD = ["LSTM"] 
DL_3D_CNN = ["CNN"] 

@st.cache_resource
def load_all_assets():
    models = {}
    # ... (Loading logic for scaler, ML, and DL models - Tetap Sama) ...
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
        except FileNotFoundError:
            st.warning(f"Model {name} not found at {path}. Skipping.")
        except Exception as e:
            st.warning(f"Error loading {name}: {e}. Skipping.")

    return models, scaler

MODELS, SCALER = load_all_assets()


# ==============================================================================
# 2. FUNGSI PREPROCESSING DAN PREDIKSI (Logika Tetap Sama)
# ==============================================================================

def preprocess_input(input_data, scaler):
    """Mengambil input dari form, memfilter HANYA 5 FITUR, dan mengembalikan DataFrame dan Array."""
    
    full_input_df = pd.DataFrame([input_data])
    input_df_filtered = full_input_df[FEATURES_USED]
    X_scaled = scaler.transform(input_df_filtered)
    
    return input_df_filtered, X_scaled 


def predict_score(model_name, model, input_df, X_scaled):
    """Melakukan prediksi, menyesuaikan reshape untuk DL (LSTM/CNN) atau menggunakan 2D (DNN/ML)."""

    if model_name in DL_3D_STANDARD: 
        X_final = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    elif model_name in DL_3D_CNN:
        X_final = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
    else: 
        if model_name in ML_PATHS:
             X_final = input_df 
        else: # DNN
             X_final = X_scaled 

    try:
        prediction = model.predict(X_final, verbose=0)
    except Exception as e:
        # Mengembalikan None jika prediksi gagal
        st.error(f"[{model_name}] Prediksi Gagal: {type(e).__name__}. Cek log error.")
        return None
    
    if model_name in DL_PATHS:
        return float(prediction[0][0]) 
    else:
        return float(prediction[0])

# ==============================================================================
# 3. STRUKTUR ANTARMUKA STREAMLIT (FINAL INTERAKTIF)
# ==============================================================================

st.set_page_config(page_title="Prediksi Nilai Siswa", layout="wide", initial_sidebar_state="auto")
st.title("💡 Form Interaktif: Prediksi Nilai Ujian Siswa")
st.markdown("---")

st.info("Input di Bagian 2 (Data Pendukung) digunakan untuk pengumpulan data, tetapi **TIDAK MEMPENGARUHI** prediksi model saat ini (yang hanya menggunakan 5 faktor utama dari Bagian 1).")

# --- BAGIAN INPUT FORM ---
st.header("Bagian 1: Kebiasaan Belajar & Kondisi Utama")
col_main_1, col_main_2, col_main_3 = st.columns(3)

# 1. Study Hours & Simple Choice
with col_main_1:
    study_hours = st.number_input(
        "Jam Belajar per Hari",
        min_value=0.0, max_value=8.0, value=4.0, step=0.5,
        help="Input antara 0 hingga 8 jam."
    )
    gender = st.radio("Jenis Kelamin", CATEGORICAL_OPTIONS['gender'], horizontal=True)
    
# 2. Attendance & Rating
with col_main_2:
    attendance = st.number_input(
        "Persentase Kehadiran (%)",
        min_value=60.0, max_value=100.0, value=90.0, step=0.1,
        help="Input antara 60% hingga 100%."
    )
    mental_health = st.radio(
        "Rating Kesehatan Mental (1-10)",
        options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], index=6, horizontal=True,
        help="Pilih rating Anda. 1 = Sangat Buruk, 10 = Sangat Baik."
    )

# 3. Sleep & Exercise
with col_main_3:
    sleep_hours = st.slider(
        "Jam Tidur per Hari",
        min_value=4.0, max_value=10.0, value=7.0, step=0.5,
        help="Geser untuk memilih jam tidur rata-rata harian Anda."
    )
    exercise_freq = st.selectbox(
        "Frekuensi Olahraga per Minggu",
        options=list(range(0, 8)), index=3,
        help="Pilih 0 hingga 7 hari."
    )


st.markdown("---")


# --- BAGIAN 2: LINGKUNGAN & DATA PENDUKUNG (DIABAIKAN MODEL) ---
st.header("Bagian 2: Data Latar Belakang (Diabaikan Model)")
col_add_1, col_add_2 = st.columns(2)

with col_add_1:
    diet_quality = st.selectbox("Kualitas Diet", CATEGORICAL_OPTIONS['diet_quality'])
    part_time_job = st.radio("Kerja Paruh Waktu", CATEGORICAL_OPTIONS['part_time_job'], horizontal=True)

with col_add_2:
    parental_education_level = st.selectbox("Edukasi Orang Tua", CATEGORICAL_OPTIONS['parental_education_level'])
    age = st.number_input("Usia (Tahun)", min_value=16, max_value=30, value=20, help="Hanya menerima angka.", key='age_input')


st.markdown("---")

# Kumpulkan data input LENGKAP dari FORM
input_data = {
    # 5 Fitur Utama (Urutan sesuai FEATURES_USED)
    "study_hours_per_day": study_hours,
    "attendance_percentage": attendance,
    "mental_health_rating": mental_health,
    "sleep_hours": sleep_hours,
    "exercise_frequency": exercise_freq,
    
    # Fitur Pendukung (Nilai ini diabaikan oleh model yang sedang dipakai)
    'gender': gender, 
    'part_time_job': part_time_job,
    'diet_quality': diet_quality,
    'parental_education_level': parental_education_level,
    'internet_quality': st.selectbox("Kualitas Internet [Diabaikan]", CATEGORICAL_OPTIONS['internet_quality'], key='internet_input'),
    'extracurricular_participation': st.radio("Ikut Ekstrakurikuler [Diabaikan]", CATEGORICAL_OPTIONS['extracurricular_participation'], horizontal=True, key='extra_input'),
    'age': age,
    'social_media_hours': st.number_input("Jam Medsos per Hari [Diabaikan]", 0.0, 6.0, 2.0, step=0.5, key='socmed_input'),
    'netflix_hours': st.slider("Jam Netflix per Hari [Diabaikan]", 0.0, 4.0, 1.0, 0.1, key='netflix_input')
}


# 4. Tombol Prediksi dan Output
st.markdown("## 📊 Hasil Prediksi dan Perbandingan")

# Tombol Aksi yang Jelas dan Terpisah
if st.button("Hitung Prediksi Semua Model", type="primary"):
    
    if not MODELS:
        st.error("Tidak ada model yang berhasil dimuat. Cek path file di terminal Anda.")
    else:
        # Preprocessing input
        input_df, X_scaled = preprocess_input(input_data, SCALER)
        
        results = []
        
        # LOOP MELALUI SEMUA MODEL DAN KUMPULKAN HASIL
        for name, model in MODELS.items():
            prediction = predict_score(name, model, input_df, X_scaled)
            
            if prediction is not None:
                results.append({
                    "Algoritma": name,
                    "Tipe": "DL" if name in DL_PATHS else "ML",
                    "Prediksi Nilai": f"{prediction:.2f}"
                })

        # Tampilkan Hasil Perbandingan
        if results:
            results_df = pd.DataFrame(results)
            # Konversi kolom nilai ke numerik untuk sorting
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
