import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error


st.set_page_config(page_title="Prediksi Harga Mobil", layout="wide", page_icon="🚗")

# ====== LOAD DATA & MODEL ======
data = pd.read_csv("data/data.csv", sep=";")

with open("models/svr_model.pkl", "rb") as f:
    pipeline = pickle.load(f)

with open("models/y_pred.pkl", "rb") as f:
    scaler_y = pickle.load(f)

data.columns = ['merek', 'model', 'daerah', 'transmisi', 'warna', 'jenis', 'tahun', 'kilometer', 'engine_size', 'harga']
data['harga'] = data['harga'].astype(str).str.replace(".", "", regex=False).astype(float)
data['kilometer'] = data['kilometer'].astype(str).str.replace(".", "", regex=False).astype(float)

cat_cols = ['merek', 'model', 'daerah', 'transmisi', 'warna', 'jenis', 'tahun', 'engine_size']
for col in cat_cols:
    data[col] = data[col].astype(str)

# ====== MENU NAVIGATION ======
st.sidebar.title("Menu")
menu = st.sidebar.radio(
    "Pilih Halaman",
    ("Beranda", "Visualisasi", "Prediksi", "Evaluasi"),
    index=0
)

# ====== BERANDA ======
if menu == "Beranda":
    st.title("🚗 Aplikasi Prediksi Harga Mobil Bekas")
    st.subheader("""
    Selamat Datang""")
    st.markdown("""
    Aplikasi Prediksi Harga Mobil Bekas adalah sebuah sistem berbasis web yang memanfaatkan teknologi Machine Learning untuk memperkirakan harga pasar dari 
    mobil bekas berdasarkan spesifikasi dan kondisi yang dimasukkan oleh pengguna.
    
    Tujuan utama dari aplikasi ini adalah untuk membantu:
    - **Calon pembeli** dalam mengetahui kisaran harga wajar sebuah mobil berdasarkan data objektif.
    - **Penjual mobil** dalam menentukan harga jual yang kompetitif dan realistis.
    - **Dealer atau showroom** dalam melakukan analisis harga dan strategi penjualan.
    
    Aplikasi ini menggunakan algoritma Support Vector Regression (SVR), yang terbukti efektif dalam menangani data numerik 
    dan memberikan prediksi harga yang akurat meskipun dalam kondisi data yang kompleks atau tidak linear.
                    """)
    st.markdown("""
    Anda dapat menavigasi ke halaman:
    - **Visualisasi**: Lihat grafik data harga mobil.
    - **Prediksi**: Masukkan spesifikasi mobil dan dapatkan prediksi harga.
    - **Evaluasi**: Lihat performa model berdasarkan MAE, RMSE, dan MAPE.
    """)

# ====== VISUALISASI ======
elif menu == "Visualisasi":
    st.subheader("📊 Visualisasi Data")

    # Load data asli dari file CSV
    raw_df = pd.read_csv("data/data.csv", sep=";")
    raw_df['Harga'] = raw_df['Harga'].astype(str).str.replace(".", "", regex=False).astype(float)
    st.session_state.raw_df = raw_df

    with st.expander("Lihat Data Awal"):
        st.dataframe(raw_df, use_container_width=True, height=400)
        
    # Histogram Distribusi Harga dan Kilometer
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 Distribusi Harga")
        fig, ax = plt.subplots(figsize=(7, 5))
        
        sns.histplot(raw_df["Harga"], bins=30, kde=True, ax=ax)
        ax.set_title("Distribusi Harga")
        ax.set_xlabel("Harga")
        ax.set_ylabel("Frekuensi")
        
        st.pyplot(fig)
    with col2:
        st.subheader("📈 Proporsi Data")
        # Pie Chart Jenis
        model_counts = raw_df["Model"].value_counts()
        labels = model_counts.index
        sizes = model_counts.values
        percentages = sizes / sizes.sum() * 100
        formatted_labels = [f"{label}\n{pct:.1f}%" for label, pct in zip(labels, percentages)]
        fig_pie, ax_pie = plt.subplots(figsize=(8, 6))
        wedges, texts = ax_pie.pie(
            sizes,
            labels=formatted_labels,
            startangle=90,
            labeldistance=1.15,
            textprops=dict(color="black", fontsize=10)
        )
        ax_pie.axis('equal')
        st.pyplot(fig_pie)
    
    # Daftar kolom kategorikal
    categorical_cols = ["Merek", "Model", "Daerah", "Transmisi", "Warna", "Jenis"]
    encoders = {}

    # Encoding manual untuk kolom kategorikal
    for col in categorical_cols:
        le = LabelEncoder()
        raw_df[col] = le.fit_transform(raw_df[col])
        encoders[col] = le
        
    # Scatter Plot antar fitur numerik
    st.subheader("⚙️ Scatter Plot Antar Fitur")

    # Ambil semua kolom numerik setelah encoding
    numerical_features_encoded = raw_df.select_dtypes(include=["number"]).columns.tolist()

    if len(numerical_features_encoded) > 1:
        x_feature = st.selectbox("Pilih fitur untuk sumbu X", numerical_features_encoded, index=0)
        y_feature = st.selectbox("Pilih fitur untuk sumbu Y", numerical_features_encoded, index=1)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=raw_df, x=x_feature, y=y_feature, ax=ax)
        ax.set_title(f'Scatter Plot: {x_feature} vs {y_feature}')
        st.pyplot(fig)
    else:
        st.warning("Data numerik kurang dari 2 fitur setelah preprocessing, tidak dapat menampilkan scatter plot.")
    st.success("✅ Visualisasi selesai")
# ====== PREDIKSI ======
elif menu == "Prediksi":
    st.title("🔍 Prediksi Harga Mobil")

    merek_to_model = {
    'Toyota': ['Avanza', 'Veloz', 'Agya', 'Calya', 'Raize', 'Yaris', 'Rush'],
    'Daihatsu': ['Ayla', 'Sigra', 'Xenia'],
    'Honda': ['Brio', 'BRV', 'CRV', 'HRV'],
    'Suzuki': ['Ertiga', 'Ignis', 'Baleno', 'XL7'],
    'Mitsubishi': ['Xpander', 'Xpander Cross']
    }
    model_to_warna = {
        'Avanza': ['Hitam', 'Silver', 'Abu-Abu', 'Putih'],
        'Veloz': ['Hitam', 'Silver', 'Putih'],
        'Agya': ['Putih', 'Abu-Abu', 'Hitam', 'Kuning', 'Merah', 'Silver'],
        'Calya': ['Silver', 'Merah', 'Abu-Abu', 'Orange', 'Hitam', 'Putih'],
        'Raize': ['Putih', 'Biru', 'Hitam', 'Merah', 'Kuning'],
        'Yaris': ['Merah', 'Putih', 'Kuning', 'Hitam'],
        'Rush': ['Hitam', 'Silver', 'Putih'],
        'Ayla': ['Putih', 'Merah', 'Hitam', 'Abu-Abu', 'Kuning', 'Silver', 'Orange'],
        'Sigra': ['Putih', 'Abu-Abu', 'Orange', 'Silver', 'Hitam'],
        'Xenia': ['Hitam', 'Abu-Abu', 'Putih'],
        'Brio': ['Hitam', 'Putih', 'Silver', 'Merah', 'Abu-Abu'],
        'BRV': ['Hitam', 'Putih', 'Abu-Abu', 'Silver'],
        'CRV': ['Hitam', 'Putih', 'Abu-Abu', 'Silver'],
        'HRV': ['Hitam', 'Putih', 'Abu-Abu', 'Silver'],
        'Ertiga': ['Hitam', 'Putih', 'Abu-Abu', 'Silver'],
        'Ignis': ['Abu-Abu', 'Putih', 'Silver', 'Merah', 'Biru', 'Hitam', 'Orange'],
        'Baleno': ['Abu-Abu', 'Merah', 'Putih', 'Hitam', 'Silver', 'Biru'],
        'XL7': ['Hitam', 'Putih', 'Abu-Abu', 'Orange', 'Coklat', 'Silver'],
        'Xpander': ['Silver', 'Putih', 'Merah', 'Hitam', 'Abu-Abu', 'Coklat'],
        'Xpander Cross': ['Hitam', 'Putih', 'Silver', 'Orange', 'Abu-Abu']
    }
    model_to_jenis = {
        'Avanza': 'MPV', 'Veloz': 'MPV', 'Agya': 'Hatchback', 'Calya': 'MPV', 'Raize': 'SUV',
        'Yaris': 'Hatchback', 'Rush': 'SUV', 'Ayla': 'Hatchback', 'Sigra': 'MPV', 'Xenia': 'MPV',
        'Brio': 'MPV', 'BRV': 'MPV', 'CRV': 'SUV', 'HRV': 'SUV', 'Ertiga': 'MPV', 'Ignis': 'MPV',
        'Baleno': 'Hatchback', 'XL7': 'MPV', 'Xpander': 'MPV', 'Xpander Cross': 'MPV'
    }
    model_to_tahun = {
        'Avanza': ['2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024'],
        'Veloz': ['2020', '2021', '2022', '2023'],
        'Agya': ['2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024'],
        'Calya': ['2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024'],
        'Raize': ['2021', '2022', '2023', '2024'],
        'Yaris': ['2021', '2022', '2023'],
        'Rush': ['2019', '2020', '2021', '2022', '2023'],
        'Ayla': ['2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024'],
        'Sigra': ['2018', '2019', '2020', '2021', '2022', '2023'],
        'Xenia': ['2019', '2020', '2021', '2022', '2023'],
        'Brio': ['2017', '2018', '2019', '2020', '2021', '2022', '2023'],
        'BRV': ['2017', '2018', '2019', '2020', '2021', '2022', '2023'],
        'CRV': ['2017', '2018', '2019', '2020', '2021', '2022', '2023'],
        'HRV': ['2017', '2018', '2019', '2020', '2021', '2022', '2023'],
        'Ertiga': ['2017', '2018', '2019', '2020', '2021', '2022', '2023'],
        'Ignis': ['2017', '2018', '2019', '2020', '2021', '2022', '2023'],
        'Baleno': ['2017', '2018', '2019', '2020', '2021', '2022', '2023'],
        'XL7': ['2020', '2021', '2022', '2023'],
        'Xpander': ['2017', '2018', '2019', '2020', '2021', '2022', '2023'],
        'Xpander Cross': ['2019', '2020', '2021', '2022', '2023']
    }
    model_to_engine = {
        'Avanza': ['1.5', '1.3'], 'Veloz': ['1.5'], 'Agya': ['1.2'], 'Calya': ['1.2'],
        'Raize': ['1.0'], 'Yaris': ['1.2'], 'Rush': ['1.5'], 'Ayla': ['1.2'], 'Sigra': ['1.2'],
        'Xenia': ['1.5'], 'Brio': ['1.2'], 'BRV': ['1.5'], 'CRV': ['1.5'], 'HRV': ['1.5'],
        'Ertiga': ['1.5'], 'Ignis': ['1.2'], 'Baleno': ['1.5'], 'XL7': ['1.5'],
        'Xpander': ['1.5'], 'Xpander Cross': ['1.5']
    }
    # Input user
    merek = st.selectbox("Merek", list(merek_to_model.keys()))
    model_mobil = st.selectbox("Model", merek_to_model[merek])
    tahun = st.selectbox("Tahun", model_to_tahun.get(model_mobil, ['2023']))
    engine = st.selectbox("Engine Size", model_to_engine.get(model_mobil, ['1.5']))
    kilometer = st.slider("Kilometer", 1, 150000, 50000)
    daerah = st.selectbox("Daerah", ['Jawa Barat', 'Banten', 'Jakarta'])
    transmisi = st.selectbox("Transmisi", ['Manual', 'Automatic'])
    warna = st.selectbox("Warna", model_to_warna.get(model_mobil, ['Hitam']))
    jenis = st.selectbox("Jenis", model_to_jenis.get(model_mobil, ['MPV']))

    input_df = pd.DataFrame([[merek, model_mobil, daerah, transmisi, warna, jenis, tahun, kilometer, engine]],
        columns=cat_cols + ['kilometer'])

    if st.button("Prediksi"):
        input_df = pd.DataFrame([[merek, model_mobil, daerah, transmisi, warna, jenis, tahun, engine, kilometer]],
            columns=cat_cols + ['kilometer'])
        
        # Prediksi dengan pipeline
        X_user = input_df.copy()
        y_pred_scaled = pipeline.predict(X_user)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))[0, 0]

        st.subheader("💰 Prediksi Harga:")
        st.write(f"Rp {int(y_pred):,}".replace(",", "."))

# ====== EVALUASI ======
elif menu == "Evaluasi":
    st.title("📈 Evaluasi Model")

    # Persiapkan data fitur dan target
    X = data.drop('harga', axis=1)
    y = data['harga']

    # Prediksi
    y_scaled = scaler_y.transform(y.values.reshape(-1, 1)).ravel()
    y_pred_scaled = pipeline.predict(X)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    # Evaluasi metrik
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mape = mean_absolute_percentage_error(y, y_pred) * 100

    # Tampilkan metrik evaluasi
    st.write(f"**Mean Absolute Error (MAE):** Rp {round(mae):,}".replace(",", "."))
    st.write(f"**Root Mean Squared Error (RMSE):** Rp {round(rmse):,}".replace(",", "."))
    st.write(f"**Mean Absolute Percentage Error (MAPE):** {mape:.2f}%")
