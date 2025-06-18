#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Tingkat Obesitas",
    page_icon="⚖️",
    layout="wide"
)

st.title("🏥 Aplikasi Prediksi Tingkat Obesitas")
st.markdown("---")

# Sidebar untuk navigasi
st.sidebar.title("📋 Menu Navigasi")
menu = st.sidebar.selectbox(
    "Pilih Menu:",
    ["🎯 Prediksi Obesitas", "ℹ️ Informasi"]
)

@st.cache_data
def load_and_prepare_data():
    """Load dan preprocessing data"""
    try:
        # Load data
        df = pd.read_csv('ObesityDataSet.csv')
        
        # Handle missing values
        df = df.replace('?', np.nan)
        kolom_numerik = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
        
        for kol in kolom_numerik:
            df[kol] = pd.to_numeric(df[kol], errors='coerce')
            if df[kol].isnull().any():
                df[kol] = df[kol].fillna(df[kol].median())
        
        # Handle categorical missing values
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mode()[0])
        
        # Data cleaning
        df['Age'] = df['Age'].round().astype(int)
        df['Height'] = df['Height'].clip(lower=0, upper=300)
        df['Weight'] = df['Weight'].round().clip(lower=0, upper=500).astype(int)
        df['FCVC'] = df['FCVC'].clip(lower=1, upper=3)
        df['NCP'] = df['NCP'].clip(lower=1, upper=4)
        df['CH2O'] = df['CH2O'].clip(lower=1, upper=3)
        df['FAF'] = df['FAF'].clip(lower=0, upper=3)
        df['TUE'] = df['TUE'].clip(lower=0, upper=2)
        
        # Remove duplicates
        df = df.drop_duplicates(keep='first').reset_index(drop=True)
        
        # Lowercase categorical columns
        for col in categorical_cols:
            df[col] = df[col].str.lower()
        
        return df
    
    except FileNotFoundError:
        st.error("File 'ObesityDataSet.csv' tidak ditemukan. Pastikan file ada di direktori yang sama.")
        return None

@st.cache_resource
def train_model(df):
    """Train model dan return model, encoder, scaler"""
    if df is None:
        return None, None, None, None
    
    # Encoding
    df_encoded = df.copy()
    
    # Binary encoding
    binary_cols = ['Gender', 'FAVC', 'SCC', 'SMOKE', 'family_history_with_overweight']
    label_encoders = {}
    
    for col in binary_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le
    
    # One-hot encoding
    non_binary_cols = ['CALC', 'CAEC', 'MTRANS']
    df_encoded = pd.get_dummies(df_encoded, columns=non_binary_cols, 
                               prefix=non_binary_cols, dtype=int)
    
    # Target encoding
    target_encoder = LabelEncoder()
    df_encoded['NObeyesdad'] = target_encoder.fit_transform(df_encoded['NObeyesdad'])
    
    # Feature selection (top 10 features based on importance)
    X = df_encoded.drop('NObeyesdad', axis=1)
    y = df_encoded['NObeyesdad']
    
    # Quick feature importance
    temp_model = RandomForestClassifier(random_state=42, n_estimators=50)
    temp_model.fit(X, y)
    importance = pd.Series(temp_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    top_10_features = importance.nlargest(10).index.tolist()
    
    # Select top features
    X_selected = X[top_10_features]
    
    # Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)
    
    # Train final model
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Best model from hyperparameter tuning
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Test accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, label_encoders, target_encoder, scaler, top_10_features, accuracy, df_encoded.columns.tolist()

def get_user_input():
    """Fungsi untuk mendapatkan input dari user"""
    st.subheader("📝 Masukkan Data Anda")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 👤 **Data Demografis**")
        gender = st.selectbox("Jenis Kelamin:", ["male", "female"])
        age = st.slider("Usia (tahun):", 14, 100, 25)
        height = st.slider("Tinggi Badan (cm):", 140, 220, 170)
        weight = st.slider("Berat Badan (kg):", 30, 200, 70)
        
        st.markdown("### 🏠 **Riwayat Keluarga**")
        family_history = st.selectbox(
            "Riwayat Keluarga dengan Kelebihan Berat Badan:", 
            ["yes", "no"]
        )
        
    with col2:
        st.markdown("### 🍽️ **Kebiasaan Makan**")
        favc = st.selectbox("Sering Makan Makanan Berkalori Tinggi:", ["yes", "no"])
        fcvc = st.slider("Frekuensi Konsumsi Sayuran (1-3):", 1.0, 3.0, 2.0, 0.1)
        ncp = st.slider("Jumlah Makan Utama (1-4):", 1.0, 4.0, 3.0, 0.1)
        caec = st.selectbox(
            "Konsumsi Makanan Antar Waktu Makan:",
            ["no", "sometimes", "frequently", "always"]
        )
        
        st.markdown("### 💧 **Gaya Hidup**")
        ch2o = st.slider("Konsumsi Air per Hari (1-3 litar):", 1.0, 3.0, 2.0, 0.1)
        scc = st.selectbox("Monitor Konsumsi Kalori:", ["yes", "no"])
        
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("### 🚬 **Kebiasaan Lain**")
        smoke = st.selectbox("Merokok:", ["yes", "no"])
        calc = st.selectbox(
            "Konsumsi Alkohol:",
            ["no", "sometimes", "frequently", "always"]
        )
        
    with col4:
        st.markdown("### 🏃 **Aktivitas Fisik**")
        faf = st.slider("Frekuensi Aktivitas Fisik (0-3):", 0.0, 3.0, 1.0, 0.1)
        tue = st.slider("Waktu Menggunakan Teknologi (0-2 jam):", 0.0, 2.0, 1.0, 0.1)
        mtrans = st.selectbox(
            "Mode Transportasi Utama:",
            ["automobile", "motorbike", "bike", "public_transportation", "walking"]
        )
    
    # Kembalikan data sebagai dictionary
    user_data = {
        'Gender': gender,
        'Age': age,
        'Height': height,
        'Weight': weight,
        'family_history_with_overweight': family_history,
        'FAVC': favc,
        'FCVC': fcvc,
        'NCP': ncp,
        'CAEC': caec,
        'CH2O': ch2o,
        'SCC': scc,
        'SMOKE': smoke,
        'CALC': calc,
        'FAF': faf,
        'TUE': tue,
        'MTRANS': mtrans
    }
    
    return user_data

def preprocess_user_input(user_data, label_encoders, all_columns, top_10_features):
    """Preprocess input user untuk prediksi"""
    # Convert to DataFrame
    df_user = pd.DataFrame([user_data])
    
    # Binary encoding menggunakan encoder yang sudah dilatih
    binary_cols = ['Gender', 'FAVC', 'SCC', 'SMOKE', 'family_history_with_overweight']
    for col in binary_cols:
        if col in label_encoders:
            try:
                df_user[col] = label_encoders[col].transform(df_user[col])
            except ValueError:
                # Jika ada nilai yang tidak dikenal, gunakan nilai default
                df_user[col] = 0
    
    # One-hot encoding untuk kolom kategorikal
    non_binary_cols = ['CALC', 'CAEC', 'MTRANS']
    df_user = pd.get_dummies(df_user, columns=non_binary_cols, 
                            prefix=non_binary_cols, dtype=int)
    
    # Pastikan semua kolom yang diperlukan ada
    for col in all_columns:
        if col not in df_user.columns and col != 'NObeyesdad':
            df_user[col] = 0
    
    # Pilih hanya kolom yang diperlukan dan urutkan sesuai training
    available_features = [col for col in top_10_features if col in df_user.columns]
    df_user = df_user[available_features]
    
    return df_user

def predict_obesity(user_data, model, label_encoders, target_encoder, scaler, top_10_features, all_columns):
    """Prediksi tingkat obesitas"""
    # Preprocess input
    processed_data = preprocess_user_input(user_data, label_encoders, all_columns, top_10_features)
    
    # Standardization
    processed_data_scaled = scaler.transform(processed_data)
    
    # Prediksi
    prediction = model.predict(processed_data_scaled)[0]
    prediction_proba = model.predict_proba(processed_data_scaled)[0]
    
    # Decode prediksi
    obesity_level = target_encoder.inverse_transform([prediction])[0]
    
    return obesity_level, prediction_proba, target_encoder.classes_

def get_obesity_info(obesity_level):
    """Informasi tentang tingkat obesitas"""
    info_dict = {
        'insufficient_weight': {
            'name': 'Berat Badan Kurang',
            'description': 'Berat badan Anda berada di bawah normal. Pertimbangkan untuk menambah asupan nutrisi yang sehat.',
            'recommendation': '• Konsultasi dengan ahli gizi\n• Tingkatkan asupan kalori sehat\n• Latihan untuk menambah massa otot',
            'color': '#3498db'
        },
        'normal_weight': {
            'name': 'Berat Badan Normal',
            'description': 'Selamat! Berat badan Anda berada dalam rentang yang sehat.',
            'recommendation': '• Pertahankan pola makan sehat\n• Rutin berolahraga\n• Pantau berat badan secara berkala',
            'color': '#2ecc71'
        },
        'overweight_level_i': {
            'name': 'Kelebihan Berat Badan Tingkat I',
            'description': 'Berat badan Anda sedikit di atas normal. Mulai perhatikan pola makan dan aktivitas fisik.',
            'recommendation': '• Kurangi porsi makan\n• Tingkatkan aktivitas fisik\n• Batasi makanan berkalori tinggi',
            'color': '#f39c12'
        },
        'overweight_level_ii': {
            'name': 'Kelebihan Berat Badan Tingkat II',
            'description': 'Berat badan Anda cukup di atas normal. Diperlukan perubahan gaya hidup yang lebih serius.',
            'recommendation': '• Konsultasi dengan dokter\n• Program diet terstruktur\n• Olahraga teratur minimal 150 menit/minggu',
            'color': '#e67e22'
        },
        'obesity_type_i': {
            'name': 'Obesitas Tingkat I',
            'description': 'Anda mengalami obesitas tingkat I. Segera ambil langkah untuk menurunkan berat badan.',
            'recommendation': '• Konsultasi dengan dokter dan ahli gizi\n• Program penurunan berat badan supervised\n• Pantau kondisi kesehatan secara rutin',
            'color': '#e74c3c'
        },
        'obesity_type_ii': {
            'name': 'Obesitas Tingkat II',
            'description': 'Anda mengalami obesitas tingkat II. Kondisi ini memerlukan penanganan medis yang serius.',
            'recommendation': '• Konsultasi dokter segera\n• Program penurunan berat badan intensif\n• Pemeriksaan kesehatan komprehensif',
            'color': '#c0392b'
        },
        'obesity_type_iii': {
            'name': 'Obesitas Tingkat III (Ekstrem)',
            'description': 'Anda mengalami obesitas ekstrem. Segera cari bantuan medis profesional.',
            'recommendation': '• Konsultasi dokter spesialis segera\n• Evaluasi untuk tindakan medis/bedah\n• Pemantauan ketat kondisi kesehatan',
            'color': '#8b0000'
        }
    }
    
    return info_dict.get(obesity_level, {
        'name': 'Tidak Dikenal',
        'description': 'Tingkat obesitas tidak dapat ditentukan.',
        'recommendation': 'Silakan konsultasi dengan dokter.',
        'color': '#95a5a6'
    })

# Menu Prediksi Obesitas
if menu == "🎯 Prediksi Obesitas":
    st.header("🎯 Prediksi Tingkat Obesitas Anda")
    
    # Load dan train model
    with st.spinner("⏳ Memuat dan melatih model..."):
        df = load_and_prepare_data()
        if df is not None:
            model, label_encoders, target_encoder, scaler, top_10_features, accuracy, all_columns = train_model(df)
            
            if model is not None:
                st.success(f"✅ Model berhasil dilatih dengan akurasi: {accuracy:.2%}")
                
                # Input form
                user_data = get_user_input()
                
                # Tombol prediksi
                if st.button("🔮 Prediksi Tingkat Obesitas", type="primary"):
                    with st.spinner("🤔 Menganalisis data Anda..."):
                        try:
                            obesity_level, prediction_proba, classes = predict_obesity(
                                user_data, model, label_encoders, target_encoder, 
                                scaler, top_10_features, all_columns
                            )
                            
                            # Tampilkan hasil
                            obesity_info = get_obesity_info(obesity_level)
                            
                            st.markdown("---")
                            st.subheader("📊 Hasil Prediksi")
                            
                            # Card hasil utama
                            st.markdown(f"""
                            <div style="
                                background-color: {obesity_info['color']}; 
                                padding: 20px; 
                                border-radius: 10px; 
                                color: white; 
                                text-align: center;
                                margin: 10px 0;
                            ">
                                <h2 style="margin: 0; color: white;">🎯 {obesity_info['name']}</h2>
                                <p style="margin: 10px 0; font-size: 16px;">{obesity_info['description']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Probabilitas prediksi
                            st.subheader("📈 Tingkat Kepercayaan Prediksi")
                            
                            # Bar chart probabilitas
                            prob_df = pd.DataFrame({
                                'Kategori': [get_obesity_info(cls)['name'] for cls in classes],
                                'Probabilitas': prediction_proba * 100
                            }).sort_values('Probabilitas', ascending=False)
                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            bars = ax.barh(prob_df['Kategori'], prob_df['Probabilitas'])
                            ax.set_xlabel('Probabilitas (%)')
                            ax.set_title('Distribusi Probabilitas Prediksi')
                            
                            # Warna bar berdasarkan nilai
                            for i, (bar, prob) in enumerate(zip(bars, prob_df['Probabilitas'])):
                                if i == 0:  # Prediksi tertinggi
                                    bar.set_color('#e74c3c')
                                else:
                                    bar.set_color('#95a5a6')
                                
                                # Label probabilitas
                                ax.text(prob + 1, bar.get_y() + bar.get_height()/2, 
                                       f'{prob:.1f}%', va='center')
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                            
                            # Rekomendasi
                            st.subheader("💡 Rekomendasi")
                            st.info(obesity_info['recommendation'])
                            
                            # BMI Calculator
                            bmi = user_data['Weight'] / ((user_data['Height']/100) ** 2)
                            st.subheader("⚖️ Informasi BMI Anda")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("BMI", f"{bmi:.1f}")
                            with col2:
                                if bmi < 18.5:
                                    bmi_category = "Underweight"
                                elif bmi < 25:
                                    bmi_category = "Normal"
                                elif bmi < 30:
                                    bmi_category = "Overweight"
                                else:
                                    bmi_category = "Obese"
                                st.metric("Kategori BMI", bmi_category)
                            with col3:
                                ideal_weight = 22.5 * ((user_data['Height']/100) ** 2)
                                st.metric("Berat Ideal", f"{ideal_weight:.1f} kg")
                            
                        except Exception as e:
                            st.error(f"❌ Terjadi kesalahan dalam prediksi: {str(e)}")
            else:
                st.error("❌ Gagal melatih model.")
        else:
            st.error("❌ Gagal memuat data.")

# Menu Analisis Data
elif menu == "📊 Analisis Data":
    st.header("📊 Analisis Dataset Obesitas")
    
    df = load_and_prepare_data()
    if df is not None:
        st.subheader("📈 Statistik Dataset")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Data", len(df))
        with col2:
            st.metric("Jumlah Fitur", len(df.columns))
        with col3:
            st.metric("Data Duplikat", df.duplicated().sum())
        with col4:
            st.metric("Missing Values", df.isnull().sum().sum())
        
        # Distribusi target
        st.subheader("🎯 Distribusi Tingkat Obesitas")
        fig, ax = plt.subplots(figsize=(12, 6))
        obesity_counts = df['NObeyesdad'].value_counts()
        
        colors = ['#3498db', '#2ecc71', '#f39c12', '#e67e22', '#e74c3c', '#c0392b', '#8b0000']
        bars = ax.bar(range(len(obesity_counts)), obesity_counts.values, color=colors)
        ax.set_xticks(range(len(obesity_counts)))
        ax.set_xticklabels([get_obesity_info(cat)['name'] for cat in obesity_counts.index], rotation=45, ha='right')
        ax.set_ylabel('Jumlah')
        ax.set_title('Distribusi Tingkat Obesitas dalam Dataset')
        
        # Label pada bar
        for bar, count in zip(bars, obesity_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                   str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Korelasi fitur numerik
        st.subheader("🔗 Korelasi Antar Fitur Numerik")
        numeric_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
        correlation_matrix = df[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, ax=ax)
        ax.set_title('Matriks Korelasi Fitur Numerik')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# Menu Informasi
elif menu == "ℹ️ Informasi":
    st.header("ℹ️ Informasi Aplikasi")
    
    st.markdown("""
    ## 🎯 Tentang Aplikasi
    
    Aplikasi **Prediksi Tingkat Obesitas** ini menggunakan teknik Machine Learning untuk memprediksi 
    tingkat obesitas seseorang berdasarkan berbagai faktor seperti:
    
    - 👤 **Data Demografis**: Usia, jenis kelamin, tinggi, berat badan
    - 🍽️ **Kebiasaan Makan**: Frekuensi konsumsi makanan berkalori tinggi, sayuran, dll.
    - 🏃 **Aktivitas Fisik**: Frekuensi olahraga dan aktivitas fisik
    - 🚗 **Gaya Hidup**: Mode transportasi, kebiasaan merokok, konsumsi alkohol
    
    ## 🤖 Model Machine Learning
    
    Aplikasi ini menggunakan **Random Forest Classifier** yang telah dioptimasi dengan:
    - ✅ Akurasi tinggi (>93%)
    - ✅ Hyperparameter tuning
    - ✅ Feature selection untuk fitur terpenting
    - ✅ Data preprocessing yang komprehensif
    
    ## 📊 Kategori Obesitas
    
    Model dapat memprediksi 7 kategori tingkat obesitas:
    """)
    
    categories = [
        'insufficient_weight', 'normal_weight', 'overweight_level_i',
        'overweight_level_ii', 'obesity_type_i', 'obesity_type_ii', 'obesity_type_iii'
    ]
    
    for category in categories:
        info = get_obesity_info(category)
        st.markdown(f"""
        <div style="
            background-color: {info['color']}; 
            padding: 15px; 
            border-radius: 8px; 
            color: white; 
            margin: 10px 0;
        ">
            <h4 style="margin: 0; color: white;">{info['name']}</h4>
            <p style="margin: 5px 0;">{info['description']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    ## ⚠️ Disclaimer
    
    **Penting untuk diingat:**
    - 🏥 Aplikasi ini hanya untuk tujuan edukasi dan screening awal
    - 👨‍⚕️ Tidak menggantikan konsultasi medis profesional
    - 📋 Hasil prediksi sebaiknya dikonfirmasi dengan pemeriksaan medis
    - 🎯 Selalu konsultasikan kondisi kesehatan Anda dengan dokter
    
    ## 👨‍💻 Pengembang
    
    Aplikasi ini dikembangkan sebagai bagian dari proyek analisis data kesehatan 
    menggunakan Python, Streamlit, dan Scikit-learn.
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>🏥 <strong>Aplikasi Prediksi Tingkat Obesitas</strong> | 
        Developed with ❤️ using Streamlit & Machine Learning</p>
        <p><small>⚠️ Untuk tujuan edukasi - Konsultasikan dengan dokter untuk diagnosis medis</small></p>
    </div>
    """, 
    unsafe_allow_html=True
)