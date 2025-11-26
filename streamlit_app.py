import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# ==============================================================================
# 1. SETUP ASET MODEL DAN DATA (Dibiarkan SAMA)
# ==============================================================================

# Definisi Fitur (Hanya 5 fitur numerik yang digunakan saat training)
FEATURES = [
    "study_hours_per_day", "attendance_percentage", "mental_health_rating", 
    "sleep_hours", "exercise_frequency"
]
# ... (Semua definisi path dan fungsi load_all_assets() tetap sama) ...
# ... (Asumsi Anda sudah mendefinisikan CATEGORICAL_OPTIONS di luar sini)

# Definisi OPSI KATEGORI untuk UI
CATEGORICAL_OPTIONS = {
    'gender': ['Female', 'Male'],
    'part_time_job': ['Tidak', 'Ya'],
    'diet_quality': ['Baik', 'Cukup', 'Buruk', 'Rata-rata'],
    'parental_education_level': ['Master', 'Sarjana (Bachelor)', 'SMA/Sederajat', 'Kuliah Non-Gelar', 'Doktor (PhD)'],
    'internet_quality': ['Baik', 'Rata-rata', 'Buruk'],
    'extracurricular_participation': ['Ya', 'Tidak']
}

# Memuat semua aset
# MODEL, SCALER = load_all_assets() 
# (Asumsi baris ini ada dan bekerja)

# ==============================================================================
# 2. FUNGSI PREPROCESSING DAN PREDIKSI (Dibiarkan SAMA)
# ==============================================================================
# ... (Fungsi preprocess_input dan predict_score tetap sama) ...


# ==============================================================================
# 3. STRUKTUR ANTARMUKA STREAMLIT (INTERAKTIF)
# ==============================================================================

st.set_page_config(page_title="Prediksi Nilai Siswa", layout="wide", initial_sidebar_state="auto")
st.title("💡 Form Interaktif: Prediksi Nilai Ujian Siswa")
st.markdown("---")

st.sidebar.header("⚙️ Konfigurasi Model")
# ... (Logika st.sidebar.selectbox tetap sama) ...

st.warning("⚠️ Perhatian: Model yang memprediksi saat ini HANYA menggunakan data di Bagian 1.")
st.info("Input di Bagian 2 (Lingkungan & Gaya Hidup) digunakan untuk tujuan pengumpulan data, tetapi TIDAK MEMPENGARUHI hasil prediksi.")

# --- BAGIAN 1: FAKTOR UTAMA (5 FITUR YANG DIGUNAKAN) ---
st.header("Bagian 1: Kebiasaan Belajar & Kondisi Utama")
st.subheader("Data ini adalah FAKTOR PREDITOR UTAMA.")

# Layout kolom untuk membuat tampilan padat dan menarik
col_main_1, col_main_2, col_main_3 = st.columns(3)

# 1. Study Hours
with col_main_1:
    study_hours = st.number_input(
        "Jam Belajar per Hari",
        min_value=0.0, max_value=8.0, value=4.0, step=0.5,
        help="Input antara 0 hingga 8 jam."
    )
    
    # 2. Attendance Percentage
    attendance = st.number_input(
        "Persentase Kehadiran (%)",
        min_value=50.0, max_value=100.0, value=90.0, step=0.1,
        help="Input antara 50% hingga 100%. Contoh: 95.5"
    )

# 3. Mental Health Rating
with col_main_2:
    mental_health = st.radio(
        "Rating Kesehatan Mental (1-10)",
        options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], index=6, horizontal=True,
        help="Pilih rating Anda. 1 = Sangat Buruk, 10 = Sangat Baik."
    )

# 4. Sleep Hours
with col_main_3:
    sleep_hours = st.slider(
        "Jam Tidur per Hari",
        min_value=4.0, max_value=10.0, value=7.0, step=0.5,
        help="Geser untuk memilih jam tidur rata-rata harian Anda."
    )
    
    # 5. Exercise Frequency
    exercise_freq = st.selectbox(
        "Frekuensi Olahraga per Minggu",
        options=list(range(0, 8)), index=3,
        help="Pilih 0 hingga 7 hari."
    )


st.markdown("---")


# --- BAGIAN 2: LINGKUNGAN & DATA PENDUKUNG (DIABAIKAN MODEL) ---
st.header("Bagian 2: Data Latar Belakang (Tidak Memengaruhi Prediksi)")
st.caption("Data ini dikumpulkan untuk perbaikan model di masa depan.")

col_add_1, col_add_2, col_add_3 = st.columns(3)

with col_add_1:
    # Kategori
    gender = st.radio("Jenis Kelamin", CATEGORICAL_OPTIONS['gender'], horizontal=True)
    part_time_job = st.radio("Kerja Paruh Waktu", CATEGORICAL_OPTIONS['part_time_job'], horizontal=True)
    
with col_add_2:
    # Numerik Tambahan
    age = st.number_input("Usia (Tahun)", min_value=16, max_value=30, value=20, help="Hanya menerima angka.")
    social_media_hours = st.number_input("Jam Medsos per Hari", 0.0, 6.0, 2.0, step=0.5)

with col_add_3:
    # Kategori
    diet_quality = st.selectbox("Kualitas Diet", CATEGORICAL_OPTIONS['diet_quality'])
    parental_education_level = st.selectbox("Edukasi Orang Tua", CATEGORICAL_OPTIONS['parental_education_level'])


# ... dan seterusnya untuk semua input kategori/numerik tambahan

st.markdown("---")

# Kumpulkan data input LENGKAP dari FORM (Termasuk yang diabaikan)
input_data = {
    # 5 Fitur Utama (Urutan sesuai FEATURES)
    "study_hours_per_day": study_hours,
    "attendance_percentage": attendance,
    "mental_health_rating": mental_health,
    "sleep_hours": sleep_hours,
    "exercise_frequency": exercise_freq,
    
    # Fitur Pendukung (Nilai ini diabaikan oleh model yang sedang dipakai)
    'age': age,
    'social_media_hours': social_media_hours,
    'netflix_hours': st.slider("Jam Netflix per Hari [Diabaikan]", 0.0, 4.0, 1.0, 0.1, key='netflix_input'),
    'gender': gender, 
    'part_time_job': part_time_job,
    'diet_quality': diet_quality,
    'parental_education_level': parental_education_level,
    'internet_quality': st.selectbox("Kualitas Internet [Diabaikan]", CATEGORICAL_OPTIONS['internet_quality']),
    'extracurricular_participation': st.radio("Ikut Ekstrakurikuler [Diabaikan]", CATEGORICAL_OPTIONS['extracurricular_participation'], horizontal=True, key='extra_input')
}


# 4. Tombol Prediksi dan Output
# ... (Logika button, preprocessing, dan prediksi tetap sama) ...
