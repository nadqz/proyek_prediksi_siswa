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

# Path file dan Konfigurasi
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
    'gender': ['Perempuan', 'Laki-laki'],
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
        st.error(f"Error: Scaler file not found at {SCALER_PATH}.")
        st.stop()
    
    all_paths = {**ML_PATHS, **DL_PATHS}
    for name, path in all_paths.items():
        try:
            if name in DL_PATHS:
                models[name] = load_model(path)
            else:
                models[name] = joblib.load(path)
        except Exception as e:
            st.warning(f"Error loading {name} from {path}. Skipping. Cek log error.")

    return models, scaler

MODELS, SCALER = load_all_assets()


# ==============================================================================
# 2. FUNGSI PREPROCESSING DAN PREDIKSI
# ==============================================================================

def preprocess_input(input_data, scaler):
    """Mengambil input form, memfilter 5 fitur utama, dan mengembalikan DataFrame dan Array."""
    full_input_df = pd.DataFrame([input_data])
    input_df_filtered = full_input_df[FEATURES_USED]
    X_scaled = scaler.transform(input_df_filtered)
    return input_df_filtered, X_scaled

def predict_score(model_name, model, input_df, X_scaled):
    """Melakukan prediksi, menyesuaikan input format (DataFrame/Array) dan shape (2D/3D)."""

    if model_name in DL_3D_STANDARD: 
        X_final = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    elif model_name in DL_3D_CNN:
        X_final = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
    elif model_name in ML_PATHS:
        X_final = input_df 
    else: # DNN
        X_final = X_scaled
    
    try:
        prediction = model.predict(X_final, verbose=0)
    except Exception as e:
        return None
    
    if model_name in DL_PATHS:
        return float(prediction[0][0]) 
    else:
        return float(prediction[0])

# ==============================================================================
# 3. STRUKTUR ANTARMUKA STREAMLIT (FINAL INTERAKTIF)
# ==============================================================================

st.set_page_config(page_title="Prediksi Nilai Siswa", layout="wide")
st.title("🎯 Formulir Interaktif: Prediksi Nilai Ujian Siswa")
st.caption("Jawablah pertanyaan di bawah ini untuk melihat estimasi nilai ujian Anda berdasarkan 6 model Machine Learning dan Deep Learning.")

# --- Bagian Peringatan ---
st.warning("⚠️ Perhatian: Model yang tersedia saat ini HANYA menggunakan 5 Faktor Utama dari Bagian 1.")

# --- BAGIAN 1: FAKTOR UTAMA (5 FITUR YANG DIGUNAKAN) ---
st.header("1. Faktor Utama (Prediktor Kuat)")

col_main_1, col_main_2, col_main_3 = st.columns(3)

with col_main_1:
    # PERTANYAAN 1: STUDY HOURS
    study_hours = st.number_input(
        "Berapa jam rata-rata Anda belajar per hari?", 
        min_value=0.0, max_value=8.0, value=4.0, step=0.5,
        help="Input antara 0 hingga 8 jam."
    )
    # PERTANYAAN KATEGORI (Hanya UI)
    gender = st.radio("Apa jenis kelamin Anda?", CATEGORICAL_OPTIONS['gender'], horizontal=True, key='gender_input')
with col_main_2:
    # PERTANYAAN 2: ATTENDANCE
    attendance = st.number_input(
        "Berapa persentase kehadiran Anda (%) di kelas/sesi?", 
        min_value=60.0, max_value=100.0, value=90.0, step=0.1,
        help="Input antara 60% hingga 100%."
    )
    # PERTANYAAN 3: MENTAL HEALTH RATING
    mental_health = st.radio(
        "Bagaimana rating kesehatan mental Anda saat ini (1-10)?",
        options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], index=6, horizontal=True,
        help="1 = Sangat Buruk, 10 = Sangat Baik."
    )
with col_main_3:
    # PERTANYAAN 4: SLEEP HOURS
    sleep_hours = st.slider(
        "Berapa jam rata-rata Anda tidur per hari?",
        min_value=4.0, max_value=10.0, value=7.0, step=0.5,
        help="Geser untuk memilih jam tidur rata-rata harian Anda."
    )
    # PERTANYAAN 5: EXERCISE FREQUENCY
    exercise_freq = st.selectbox(
        "Berapa kali (hari) Anda berolahraga dalam seminggu?",
        options=list(range(0, 8)), index=3,
        help="Pilih 0 hingga 7 hari."
    )


st.markdown("---")


# --- BAGIAN 2: DATA LATAR BELAKANG (DIABAIKAN MODEL) ---
st.header("2. Data Latar Belakang (Tidak Memengaruhi Prediksi)")
st.caption("Data ini hanya untuk kepentingan studi dan pengumpulan data di masa mendatang.")

col_add_1, col_add_2 = st.columns(2)

with col_add_1:
    # PERTANYAAN KATEGORI
    diet_quality = st.selectbox("Bagaimana kualitas diet Anda?", CATEGORICAL_OPTIONS['diet_quality'], key='diet')
    part_time_job = st.radio("Apakah Anda memiliki pekerjaan paruh waktu?", CATEGORICAL_OPTIONS['part_time_job'], horizontal=True, key='job_input')
with col_add_2:
    # PERTANYAAN KATEGORI
    parental_education_level = st.selectbox("Apa tingkat pendidikan tertinggi orang tua Anda?", CATEGORICAL_OPTIONS['parental_education_level'], key='parental')
    age = st.number_input("Berapa usia Anda saat ini?", min_value=16, max_value=30, value=20, key='age_input')


# Kumpulkan data input LENGKAP dari FORM
input_data = {
    "study_hours_per_day": study_hours, "attendance_percentage": attendance, "mental_health_rating": mental_health,
    "sleep_hours": sleep_hours, "exercise_frequency": exercise_freq,
    
    # Fitur Pendukung (Diabaikan)
    'gender': gender, 'part_time_job': part_time_job, 'diet_quality': diet_quality,
    'parental_education_level': parental_education_level, 
    
    # Gunakan nilai default/input untuk variabel yang ditampilkan di form
    'internet_quality': st.selectbox("Kualitas Internet di tempat tinggal Anda?", CATEGORICAL_OPTIONS['internet_quality'], key='internet_input'),
    'extracurricular_participation': st.radio("Apakah Anda ikut ekstrakurikuler?", CATEGORICAL_OPTIONS['extracurricular_participation'], horizontal=True, key='extra_input'),
    'age': age, 
    'social_media_hours': st.number_input("Berapa jam rata-rata Anda menggunakan media sosial per hari?", 0.0, 6.0, 2.0, step=0.5, key='socmed_input'),
    'netflix_hours': st.slider("Berapa jam rata-rata Anda menonton hiburan (Netflix/lainnya) per hari?", 0.0, 4.0, 1.0, 0.1, key='netflix_input')
}


# 4. Tombol Prediksi dan Output
st.markdown("## 📊 Hasil Prediksi dan Perbandingan")

# Tombol aksi tunggal
if st.button("Hitung Prediksi Semua Model", type="primary"):
    
    if not MODELS:
        st.error("Tidak ada model yang berhasil dimuat. Cek log error deployment.")
    else:
        # Preprocessing input
        input_df, X_scaled = preprocess_input(input_data, SCALER)
        results = []
        
        # LOOP MELALUI SEMUA MODEL
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
