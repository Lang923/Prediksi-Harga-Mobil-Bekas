import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_option_menu import option_menu

# ====== PAGE CONFIG ======
st.set_page_config(
    page_title="CarPrice Predictor",
    layout="wide",
    page_icon="🚗",
    initial_sidebar_state="expanded"
)

# ====== CUSTOM CSS STYLING ======
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #32b8ba;
        --secondary-color: #0a5eb7;
        --accent-color: #32b8ba;
        --background-color: #F8F9FA;
        --text-color: #2C3E50;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Custom sidebar styling */
    .css-1d391kg {
        background: linear-gradient(135deg, #32b8ba 0%, #0a5eb7 100%);
    }

    /* Main content area */
    .main-header {
        background: linear-gradient(135deg, #32b8ba 0%, #0a5eb7 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }

    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 5px solid #32b8ba;
        transition: transform 0.3s ease;
    }

    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }

    .metric-card {
        background: linear-gradient(135deg, #32b8ba 0%, #0a5eb7 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 0.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }

    .prediction-result {
        background: linear-gradient(135deg, #32b8ba 0%, #a8d8ff 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: 3px solid #32b8ba;
    }

    .stSelectbox > div > div {
        border-radius: 10px;
        border: 2px solid #32b8ba;
    }

    .stSlider > div > div {
        border-radius: 10px;
        border: 2px solid #32b8ba;
    }

    /* Navigation styling */
    .nav-item {
        padding: 0.5rem 1rem;
        margin: 0.2rem 0;
        border-radius: 10px;
        transition: all 0.3s ease;
    }

    .nav-item:hover {
        background-color: rgba(22, 139, 250, 0.1);
    }

    /* Charts styling */
    .chart-container {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 5px solid #32b8ba;
    }
</style>
""", unsafe_allow_html=True)

# ====== LOAD DATA & MODEL ======
@st.cache_data
def load_data():
    try:
        data = pd.read_csv("data/data.csv", sep=";")
        data.columns = ['merek', 'model', 'daerah', 'transmisi', 'warna', 'jenis', 'tahun', 'kilometer', 'engine_size', 'harga']
        data['harga'] = data['harga'].astype(str).str.replace(".", "", regex=False).astype(float)
        data['kilometer'] = data['kilometer'].astype(str).str.replace(".", "", regex=False).astype(float)
        
        cat_cols = ['merek', 'model', 'daerah', 'transmisi', 'warna', 'jenis', 'tahun', 'engine_size']
        for col in cat_cols:
            data[col] = data[col].astype(str)
        
        return data, cat_cols
    except:
        st.error("❌ Error loading data. Please check if data/data.csv exists.")
        return None, None

@st.cache_resource
def load_models():
    try:
        with open("models/svr_model.pkl", "rb") as f:
            pipeline = pickle.load(f)
        with open("models/y_pred.pkl", "rb") as f:
            scaler_y = pickle.load(f)
        return pipeline, scaler_y
    except:
        st.error("❌ Error loading models. Please check if model files exist.")
        return None, None

# Load data and models
data, cat_cols = load_data()
pipeline, scaler_y = load_models()

# ====== SIDEBAR NAVIGATION ======
st.sidebar.markdown("""
<h1 style="color: white; text-align: center; margin: 0; font-size: 3rem; margin-bottom: 1rem">🚗</h1>

""", unsafe_allow_html=True)

with st.sidebar:
    selected_menu = option_menu(
            menu_title="Navigasi",
            options=["🏠 Beranda", "📊 Visualisasi", "🔍 Prediksi", "📈 Evaluasi"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "10px", "background-color": "#fafafa"},
                "icon": {"color": "#4eb4b5", "font-size": "18px"},
                "nav-link": {
                    "font-size": "16px",
                    "text-align": "left",
                    "margin": "5px",
                    "--hover-color": "#ffe8d9"
                },
                "nav-link-selected": {
                    "background-color": "#4eb4b5", "color": "white"
                }
            }
        )
menu = selected_menu

# ====== BERANDA ======
if selected_menu == "🏠 Beranda":
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 3rem;">🚗 CarPrice Predictor</h1>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">
            Prediksi Harga Mobil Bekas dengan AI
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Welcome section
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🎯 Tentang Aplikasi</h3>
            <p>
                CarPrice Predictor adalah sebuah aplikasi berbasis kecerdasan buatan (AI) yang dirancang untuk membantu 
                pengguna dalam memperkirakan harga mobil bekas secara cepat dan akurat. 
                Aplikasi ini menggunakan algoritma Support Vector Regression (SVR), salah satu metode machine learning yang andal dalam memodelkan hubungan kompleks antar fitur.
            </p>
            
        </div>
        <div class="feature-card">
            <h3>🔥 Fitur Unggulan:</h3>
            <ul>
                <li>📊 <strong>Prediksi Akurat</strong> – Menggunakan algoritma machine learning terdepan</li>
                <li>📈 <strong>Visualisasi Data</strong> – Grafik interaktif untuk analisis pasar</li>
                <li>⚡ <strong>Interface Modern</strong> – Tampilan responsif dan user-friendly</li>
                <li>🎯 <strong>Multi-Platform</strong> – Dapat diakses di berbagai perangkat</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="feature-card" style="text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">
            <h3>📈 Statistik</h3>
            <div style="margin: 1rem 0;">
                <h2 style="margin: 0.5rem 0; color: #FFD700;">15+</h2>
                <p style="margin: 0;">Model Mobil</p>
            </div>
            <div style="margin: 1rem 0;">
                <h2 style="margin: 0.5rem 0; color: #FFD700;">5</h2>
                <p style="margin: 0;">Brand Populer</p>
            </div>
            <div style="margin: 1rem 0;">
                <h2 style="margin: 0.5rem 0; color: #FFD700;">96.74</h2>
                <p style="margin: 0;">Akurasi Prediksi</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Target users
    st.markdown("""
    <div class="feature-card">
        <h3>👥 Siapa yang Dapat Menggunakan?</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>🛒 Calon Pembeli</h4>
            <p>Dapatkan harga referensi sebelum membeli mobil bekas</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>💰 Penjual</h4>
            <p>Tentukan harga jual yang kompetitif dan realistis</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4>🏢 Dealer</h4>
            <p>Analisis harga pasar untuk strategi bisnis</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick navigation
    st.markdown("""
    <div class="feature-card">
        <h3>🚀 Mulai Menggunakan</h3>
        <p>Pilih menu di sidebar untuk mulai menggunakan aplikasi:</p>
    </div>
    """, unsafe_allow_html=True)

# ====== VISUALISASI ======
elif selected_menu == "📊 Visualisasi":
    st.markdown("""
    <div class="main-header">
        <h1 style="margin: 0;">📊 Visualisasi Data</h1>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Analisis mendalam data harga mobil bekas</p>
    </div>
    """, unsafe_allow_html=True)
    
    if data is not None:
        # Data overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{len(data):,}</h3>
                <p>Total Data</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{data['merek'].nunique()}</h3>
                <p>Brand Mobil</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            avg_price = data['harga'].mean()
            st.markdown(f"""
            <div class="metric-card">
                <h3>Rp {avg_price/1e6:.1f}M</h3>
                <p>Rata-rata Harga</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            avg_km = data['kilometer'].mean()
            st.markdown(f"""
            <div class="metric-card">
                <h3>{avg_km/1000:.1f}K</h3>
                <p>Rata-rata KM</p>
            </div>
            """, unsafe_allow_html=True)
        # Data table
        with st.expander("🗂️ Lihat Data Lengkap"):
            st.dataframe(
                data.head(100),
                use_container_width=True,
                height=400
            )
        # Charts
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.subheader("📈 Distribusi Harga Mobil Bekas")
            fig = px.histogram(
                data,
                x="harga",
                nbins=40,
                title="Distribusi Harga",
                color_discrete_sequence=["#FF6B35"],
                opacity=0.85,
                hover_data=data.columns
            )
            fig.update_layout(
                xaxis_title="Harga (Rupiah)",
                yaxis_title="Frekuensi",
                bargap=0.05,
                plot_bgcolor='rgba(245,245,245,1)',
                paper_bgcolor='rgba(255,255,255,1)',
                font=dict(family="Arial", size=12),
                title_font=dict(size=20, family='Arial', color='#333'),
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=13,
                    font_family="Arial"
                ),
                xaxis=dict(
                    showgrid=True,
                    gridcolor='lightgray',
                ),
                yaxis=dict(
                    showgrid=False,
                )
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.subheader("🥧 Proporsi Model")
            
            model_counts = data["model"].value_counts()
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
        
        # Scatter plot
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("⚙️ Analisis Korelasi")
        # ====== Encoding Kolom Kategorikal ======
    categorical_cols = ["merek", "model", "daerah", "transmisi", "warna", "jenis"]
    encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        encoders[col] = le

    # ====== Visualisasi Scatter Plot ======
    st.subheader("📊 Scatter Plot Antar Fitur")

    # Ambil kolom numerik (termasuk hasil encoding)
    numerical_features = data.select_dtypes(include=["number"]).columns.tolist()

    # Pilih fitur untuk sumbu X dan Y
    if len(numerical_features) >= 2:
        x_feature = st.selectbox("Pilih fitur untuk sumbu X", numerical_features, index=0)
        y_feature = st.selectbox("Pilih fitur untuk sumbu Y", numerical_features, index=1)
        color_feature = st.selectbox("Pilih fitur kategorikal untuk pewarnaan (opsional)", numerical_features, index=2)

        # ==== Scatter Plot dengan Plotly ====
        fig2 = px.scatter(
            data,
            x=x_feature,
            y=y_feature,
            color=color_feature,
            title=f"Scatter Plot Interaktif: {x_feature} vs {y_feature}",
            hover_data=data.columns,
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        fig2.update_layout(
            xaxis_title=x_feature.title(),
            yaxis_title=y_feature.title(),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("Data numerik kurang dari 2 fitur setelah encoding, tidak dapat menampilkan scatter plot.")
        

# ====== PREDIKSI ======
elif selected_menu == "🔍 Prediksi":
    st.markdown("""
    <div class="main-header">
        <h1 style="margin: 0;">🔍 Prediksi Harga Mobil</h1>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Masukkan spesifikasi mobil untuk mendapatkan estimasi harga</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Car specifications mappings (same as original)
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
    
    # Input form
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.subheader("🚗 Spesifikasi Mobil")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        merek = st.selectbox("🏷️ Merek", list(merek_to_model.keys()))
        model_mobil = st.selectbox("🚙 Model", merek_to_model[merek])
        daerah = st.selectbox("📍 Daerah", ['Jawa Barat', 'Banten', 'Jakarta'])
    
    with col2:
        tahun = st.selectbox("📅 Tahun", model_to_tahun.get(model_mobil, ['2023']))
        transmisi = st.selectbox("⚙️ Transmisi", ['Manual', 'Automatic'])
        engine = st.selectbox("🔧 Engine Size (L)", model_to_engine.get(model_mobil, ['1.5']))
    
    with col3:
        warna = st.selectbox("🎨 Warna", model_to_warna.get(model_mobil, ['Hitam']))
        jenis = st.text_input("📋 Jenis", value=model_to_jenis.get(model_mobil, 'MPV'), disabled=True)
        kilometer = st.slider("🛣️ Kilometer", 1, 150000, 50000, step=1000)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Prediction button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_btn = st.button(
            "🔮 PREDIKSI HARGA",
            use_container_width=True,
            type="primary"
        )
    
    if predict_btn and pipeline is not None and scaler_y is not None:
        try:
            # Prepare input data
            input_df = pd.DataFrame([[merek, model_mobil, daerah, transmisi, warna, jenis, tahun, engine, kilometer]],
                columns=['merek', 'model', 'daerah', 'transmisi', 'warna', 'jenis', 'tahun', 'engine_size', 'kilometer'])
            
            # Make prediction
            y_pred_scaled = pipeline.predict(input_df)
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))[0, 0]
            
            # Display result
            st.markdown(f"""
            <div class="prediction-result">
                <h2 style="color: white; margin: 0;">💰 Estimasi Harga</h2>
                <h1 style="color: white; margin: 1rem 0; font-size: 3rem;">
                    Rp {int(y_pred):,}
                </h1>
                <p style="color: white; margin: 0; opacity: 0.9;">
                    *Estimasi berdasarkan data pasar dan kondisi kendaraan
                </p>
            </div>
            """.replace(",", "."), unsafe_allow_html=True)
            
            # Additional info
            col1, col2, col3 = st.columns(3)
            
            with col1:
                min_price = y_pred * (1 - 0.03)
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #FF6B35 0%, #F7931E 100%);">
                    <h4>Harga Minimum</h4>
                    <h3>Rp {int(min_price):,}</h3>
                </div>
                """.replace(",", "."), unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);">
                    <h4>Harga Prediksi</h4>
                    <h3>Rp {int(y_pred):,}</h3>
                </div>
                """.replace(",", "."), unsafe_allow_html=True)
            
            with col3:
                max_price = y_pred * (1 + 0.03)
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                    <h4>Harga Maksimum</h4>
                    <h3>Rp {int(max_price):,}</h3>
                </div>
                """.replace(",", "."), unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"❌ Error dalam prediksi: {str(e)}")


# ====== EVALUASI ======
elif selected_menu == "📈 Evaluasi":
    st.markdown("""
    <div class="main-header">
        <h1 style="margin: 0;">📈 Evaluasi Model</h1>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Analisis performa dan akurasi model prediksi</p>
    </div>
    """, unsafe_allow_html=True)
    
    if data is not None and pipeline is not None and scaler_y is not None:
        try:
            # Prepare data for evaluation
            X = data.drop('harga', axis=1)
            y = data['harga']
            
            # Make predictions
            y_pred_scaled = pipeline.predict(X)
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
            
            # Calculate metrics
            mae = mean_absolute_error(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            mape = mean_absolute_percentage_error(y, y_pred) * 100
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>📊 MAE</h3>
                    <h2>Rp {mae/1e6:.2f}M</h2>
                    <p>Mean Absolute Error</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>📈 RMSE</h3>
                    <h2>Rp {rmse/1e6:.2f}M</h2>
                    <p>Root Mean Square Error</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🎯 MAPE</h3>
                    <h2>{mape:.2f}%</h2>
                    <p>Mean Absolute Percentage Error</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Performance visualization
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.subheader("📊 Visualisasi Performa Model")
            
            # Actual vs Predicted scatter plot
            fig = px.scatter(
                x=y, 
                y=y_pred,
                title="Actual vs Predicted Values",
                labels={'x': 'Harga Aktual', 'y': 'Harga Prediksi'},
                opacity=0.6
            )
            
            # Add perfect prediction line
            min_val = min(y.min(), y_pred.min())
            max_val = max(y.max(), y_pred.max())
            fig.add_shape(
                type="line",
                x0=min_val, y0=min_val,
                x1=max_val, y1=max_val,
                line=dict(color="red", width=2, dash="dash")
            )
            
            fig.update_layout(
                xaxis_title="Harga Aktual (Rupiah)",
                yaxis_title="Harga Prediksi (Rupiah)",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Model interpretation
            st.markdown("""
            <div class="feature-card">
                <h3>🧠 Interpretasi Model</h3>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem;">
                    <div>
                        <h4>✅ Kelebihan Model:</h4>
                        <ul>
                            <li>Akurasi tinggi dengan MAPE rendah</li>
                            <li>Mampu menangani data non-linear</li>
                            <li>Robust terhadap outliers</li>
                            <li>Generalisasi baik untuk data baru</li>
                        </ul>
                    </div>
                    <div>
                        <h4>⚠️ Catatan Penting:</h4>
                        <ul>
                            <li>Prediksi berdasarkan data historis</li>
                            <li>Kondisi pasar dapat mempengaruhi harga</li>
                            <li>Faktor externa tidak diperhitungkan</li>
                            <li>Gunakan sebagai referensi estimasi</li>
                        </ul>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Detailed metrics table
            with st.expander("📋 Detail Metrik Evaluasi"):
                metrics_df = pd.DataFrame({
                    'Metrik': ['Mean Absolute Error (MAE)', 'Root Mean Square Error (RMSE)', 'Mean Absolute Percentage Error (MAPE)'],
                    'Nilai': [f'Rp {mae:,.0f}', f'Rp {rmse:,.0f}', f'{mape:.2f}%'],
                    'Interpretasi': [
                        'Rata-rata selisih absolut antara prediksi dan aktual',
                        'Akar dari rata-rata kuadrat error (sensitif terhadap outlier)',
                        'Persentase rata-rata error relatif terhadap nilai aktual'
                    ]
                })
                st.dataframe(metrics_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Error dalam evaluasi: {str(e)}")
    else:
        st.warning("⚠️ Data atau model tidak tersedia untuk evaluasi.")

# ====== FOOTER ======
st.markdown(""" 
<div style="margin-top: 1rem; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
           border-radius: 15px; text-align: center; color: white;">
    <h3 style="margin: 0 0 1rem 0;">🚗 CarPrice Predictor</h3>
    <p style="margin: 0; opacity: 0.8;">
        Developed with Herlangga using Streamlit & Machine Learning
    </p>
    <p style="margin: 0.5rem 0 0 0; opacity: 0.6; font-size: 0.9rem;">
        © 2024 - Prediksi Harga Mobil Bekas Indonesia
    </p>
</div>
""", unsafe_allow_html=True)
