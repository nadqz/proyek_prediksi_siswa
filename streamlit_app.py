import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# ==============================================================================
# 1. SETUP ASET MODEL DAN DATA
# ==============================================================================

# Definisikan 5 Fitur yang BENAR-BENAL DIGUNAKAN OLEH MODEL
FEATURES_USED = [
    "study_hours_per_day", "attendance_percentage", "mental_health_rating", 
    "sleep_hours", "exercise_frequency"
]

# Path file dan Konfigurasi
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
    """Memuat SEMUA model (ML dan DL) yang ada, untuk diuji LIVE."""
    models = {}
    
    try:
        scaler = joblib.load('PREPROCESSOR/minmax_scaler.pkl')
    except FileNotFoundError:
        st.error(f"Error: Scaler file not found.")
        st.stop()
    
    all_paths = {**ML_PATHS, **DL_PATHS}
    for name, path in all_paths.items():
        try:
            if name in DL_PATHS:
                models[name] = load_model(path)
            else:
                models[name] = joblib.load(path) # Mencoba memuat file .pkl
        except Exception as e:
            st.warning(f"Error loading {name} from {path}. Model ini mungkin GAGAL LIVE. Log: {e}")
            models[name] = None # Jika gagal dimuat, set ke None

    return models, scaler

MODELS, SCALER = load_all_assets()


# ==============================================================================
# 2. FUNGSI PREPROCESSING DAN PREDIKSI
# ==============================================================================

def preprocess_input(input_data, scaler):
    """Mengambil input form, memfilter 5 fitur utama, dan mengembalikan Array NumPy yang sudah di-scale."""
    
    full_input_df = pd.DataFrame([input_data])
    input_df_filtered = full_input_df[FEATURES_USED]
    X_scaled = scaler.transform(input_df_filtered)
    
    return X_scaled 


def predict_score(model_name, model, X_scaled):
    """Melakukan prediksi, menyesuaikan reshape untuk DL atau menggunakan 2D (DNN/ML)."""

    # 1. Tentukan input akhir X_final
    if model_name in DL_3D_STANDARD: 
        # LSTM
        X_final = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    elif model_name in DL_3D_CNN:
        # CNN
        X_final = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
    else: 
        # DNN dan SEMUA MODEL ML (Menggunakan Array NumPy 2D)
        X_final = X_scaled
    
    # 2. Lakukan Prediksi
    try:
        prediction = model.predict(X_final, verbose=0)
    except Exception as e:
        # Jika prediksi gagal, kembalikan None
        st.error(f"Prediction Failed for {model_name}. Log: {type(e).__name__}")
        return None
    
    # 3. Ambil nilai prediksi
    if model_name in DL_PATHS:
        return float(prediction[0][0]) 
    else:
        return float(prediction[0]) 

# ==============================================================================
# 3. NAVIGASI STREAMLIT DAN HALAMAN
# ==============================================================================

st.set_page_config(page_title="Prediksi Nilai Siswa", layout="wide")
st.sidebar.title("Navigasi Algoritma")

menu_selection = st.sidebar.radio(
    "Pilih Fokus Prediksi:",
    ["Deep Learning (LIVE)", "Machine Learning (LIVE)"]
)

st.title("🎯 Proyek Benchmarking Prediksi Nilai")
st.markdown("---")


# --- DEFINISI FORM INPUT (Digunakan di kedua halaman) ---
def get_input_form():
    """Menampilkan form dan mengumpulkan input (5 fitur utama dan tambahan)."""
    
    # --- BAGIAN 1: FAKTOR UTAMA ---
    st.header("1. Faktor Utama (Prediktor Kuat)")

    col_main_1, col_main_2, col_main_3 = st.columns(3)

    with col_main_1:
        study_hours = st.number_input("Berapa jam rata-rata Anda belajar per hari?", 0.0, 8.0, 4.0, step=0.5, key='study')
        gender = st.radio("Apa jenis kelamin Anda?", CATEGORICAL_OPTIONS['gender'], horizontal=True, key='gender_input')
    with col_main_2:
        attendance = st.number_input("Berapa persentase kehadiran Anda (%) di kelas/sesi?", 60.0, 100.0, 90.0, step=0.1, key='attn')
        mental_health = st.radio("Bagaimana rating kesehatan mental Anda saat ini (1-10)?", options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], index=6, horizontal=True, key='mental')
    with col_main_3:
        sleep_hours = st.slider("Berapa jam rata-rata Anda tidur per hari?", 4.0, 10.0, 7.0, step=0.5, key='sleep')
        exercise_freq = st.selectbox("Berapa kali (hari) Anda berolahraga dalam seminggu?", options=list(range(0, 8)), index=3, key='exercise')

    st.markdown("---")
    
    # --- BAGIAN 2: DATA LATAR BELAKANG (Diabaikan Model) ---
    st.header("2. Data Latar Belakang (Tidak Memengaruhi Prediksi)")
    st.caption("Data di bagian ini hanya untuk pengumpulan dan tidak memengaruhi hasil prediksi.")

    col_add_1, col_add_2 = st.columns(2)
    with col_add_1:
        diet_quality = st.selectbox("Bagaimana kualitas diet Anda?", CATEGORICAL_OPTIONS['diet_quality'], key='diet')
        part_time_job = st.radio("Apakah Anda memiliki pekerjaan paruh waktu?", CATEGORICAL_OPTIONS['part_time_job'], horizontal=True, key='job_input')
    with col_add_2:
        parental_education_level = st.selectbox("Apa tingkat pendidikan tertinggi orang tua Anda?", CATEGORICAL_OPTIONS['parental_education_level'], key='parental')
        age = st.number_input("Berapa usia Anda saat ini?", min_value=16, max_value=30, value=20, key='age_input')


    input_data = {
        "study_hours_per_day": study_hours, "attendance_percentage": attendance, "mental_health_rating": mental_health,
        "sleep_hours": sleep_hours, "exercise_frequency": exercise_freq,
        'gender': gender, 'part_time_job': part_time_job, 'diet_quality': diet_quality,
        'parental_education_level': parental_education_level, 
        'internet_quality': st.selectbox("Kualitas Internet di tempat tinggal Anda?", CATEGORICAL_OPTIONS['internet_quality'], key='internet_input'),
        'extracurricular_participation': st.radio("Apakah Anda ikut ekstrakurikuler?", CATEGORICAL_OPTIONS['extracurricular_participation'], horizontal=True, key='extra_input'),
        'age': age, 
        'social_media_hours': st.number_input("Berapa jam rata-rata Anda menggunakan media sosial per hari?", 0.0, 6.0, 2.0, step=0.5, key='socmed_input'),
        'netflix_hours': st.slider("Berapa jam rata-rata Anda menonton hiburan (Netflix/lainnya) per hari?", 0.0, 4.0, 1.0, 0.1, key='netflix_input')
    }
    return input_data


# --- LOGIKA HALAMAN ---

if menu_selection == "Deep Learning (LIVE)":
    
    st.header("Model Deep Learning (LSTM, CNN, DNN)")
    st.info("Nilai prediksi dihitung secara *live* menggunakan model Deep Learning.")
    
    input_data = get_input_form()
    
    if st.button("Hitung Prediksi DL", type="primary"):
        X_scaled = preprocess_input(input_data, SCALER)
        results = []
        
        for name in DL_PATHS.keys():
            model = MODELS.get(name)
            if model: # Hanya jika model berhasil dimuat
                prediction = predict_score(name, model, X_scaled)
                if prediction is not None:
                    results.append({"Algoritma": name, "Prediksi Nilai": f"{prediction:.2f}"})

        if results:
            results_df = pd.DataFrame(results)
            results_df['Prediksi Nilai'] = results_df['Prediksi Nilai'].astype(float)
            st.markdown("### Hasil Prediksi Live (DL)")
            st.dataframe(results_df.sort_values(by="Prediksi Nilai", ascending=False).set_index("Algoritma"), use_container_width=True)
        else:
            st.error("Gagal mendapatkan hasil. Cek log error deployment.")

elif menu_selection == "Machine Learning (LIVE)":
    
    st.header("Model Machine Learning (RF, DT, LR)")
    st.warning("⚠️ Karena konflik versi, model ini mungkin gagal diprediksi. Harap lihat log error jika tombol ditekan.")
    
    input_data = get_input_form() 
    
    if st.button("Hitung Prediksi ML", type="primary"):
        X_scaled = preprocess_input(input_data, SCALER)
        results = []
        
        for name in ML_PATHS.keys():
            model = MODELS.get(name)
            if model and model != "STATIC_MODEL": # Hanya jika model berhasil dimuat dan bukan placeholder
                prediction = predict_score(name, model, X_scaled)
                if prediction is not None:
                    results.append({"Algoritma": name, "Prediksi Nilai": f"{prediction:.2f}"})
        
        if results:
            results_df = pd.DataFrame(results)
            results_df['Prediksi Nilai'] = results_df['Prediksi Nilai'].astype(float)
            st.markdown("### Hasil Prediksi Live (ML)")
            st.dataframe(results_df.sort_values(by="Prediksi Nilai", ascending=False).set_index("Algoritma"), use_container_width=True)
        else:
            st.error("Gagal mendapatkan hasil dari model ML. File .pkl Anda tidak kompatibel.")
