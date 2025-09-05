import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------------
# Judul aplikasi
# ---------------------------
st.title("Prediksi Sales")
st.write("Masukkan anggaran iklan untuk memprediksi jumlah produk terjual.")

# ---------------------------
# Load model & scaler
# ---------------------------
model = joblib.load("bagging_tree_terbaik.joblib")
scaler = joblib.load("scaler_X.joblib")

# ---------------------------
# Input user
# ---------------------------
st.header("Anggaran Iklan (dalam ribuan $)")
tv = st.number_input("TV ($k)", min_value=0.0, value=100.0, step=1.0)
radio = st.number_input("Radio ($k)", min_value=0.0, value=30.0, step=1.0)
newspaper = st.number_input("Newspaper ($k)", min_value=0.0, value=20.0, step=1.0)

# Transformasi cube root untuk Newspaper
newspaper_cbrt = np.cbrt(newspaper)

# Buat DataFrame dari input user
input_df = pd.DataFrame({
    'TV': [tv],
    'Radio': [radio],
    'Newspaper_cbrt': [newspaper_cbrt]
})

# ---------------------------
# Standardisasi input
# ---------------------------
input_scaled = scaler.transform(input_df)

# ---------------------------
# Prediksi
# ---------------------------
prediksi = model.predict(input_scaled)[0]

# ---------------------------
# Tampilkan hasil
# ---------------------------
st.subheader("Hasil Prediksi Sales (ribuan unit):")
st.write(f"Prediksi: **{prediksi:.2f} ribu unit**")
