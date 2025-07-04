#!/usr/bin/env python
# coding: utf-8

# In[48]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier 
import streamlit as st


# In[49]:

st.title("Capstone Project: Analisis Obesitas")

#  LOAD DATA
df = pd.read_csv('ObesityDataSet.csv')


# In[50]:


# 1. TAMPILKAN BEBERAPA BARIS PERTAMA DAN INFORMASI UMUM DATASET
st.subheader("1. EDA (Exploratory Data Analysis)")


# In[51]:


# Menampilkan 5 baris pertama
st.write("\nDATA PREVIEW (5 BARIS PERTAMA):")
st.dataframe(df.head())


# In[52]:


# Informasi umum dataset
st.write("\nINFORMASI UMUM DATASET:")
st.write(f"Jumlah baris: {df.shape[0]}")
st.write(f"Jumlah kolom: {df.shape[1]}")
st.write(f"Nama kolom: {', '.join(df.columns)}")


# In[53]:


# Tipe data
df = df.replace('?', np.nan)
kolom_numerik = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
for kol in kolom_numerik:
    df[kol] = pd.to_numeric(df[kol], errors='coerce')
st.write("\nTIPE DATA:")
st.dataframe(df.dtypes)


# In[54]:


# Deskripsi Data
st.write("\nDESKRIPSI DATA :")
st.dataframe(df.describe(include='all'))


# In[55]:


# VISUALISASI DATA
st.subheader("VISUALISASI DATA")


# In[56]:


# Pengaturan untuk tampilan grafik
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (14, 8)


# In[57]:


# Visualisasi kolom numerik
st.write("\nVisualisai KOLOM NUMERIK")
for i, col in enumerate(kolom_numerik):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    sns.histplot(data=df, x=col, kde=True)
    plt.title(f'Histogram {col}')

    plt.subplot(1, 2, 2)
    sns.boxplot(data=df, y=col)
    plt.title(f'Boxplot {col}')

    plt.tight_layout()
    file_path = os.path.join('Assets/visualisasi_data/kolom_numerik', f'distribusi_{col}.png')

    plt.savefig(file_path)
    st.pyplot(plt)
    plt.close()  # Tutup plot untuk menghemat memori


# In[58]:


# Visualisasi kolom categorical
st.write("\nVisualisai KOLOM CATEGORICAL")
for col in df.select_dtypes(include=['object']).columns:
    plt.figure(figsize=(12, 6))

    # Count plot for categorical data
    plt.subplot(1, 2, 1)
    sns.countplot(data=df, y=col)
    plt.title(f'Count Plot {col}')

    # Pie chart for percentage distribution
    plt.subplot(1, 2, 2)
    df[col].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title(f'Percentage Distribution {col}')

    plt.tight_layout()
    file_path = os.path.join('Assets/visualisasi_data/kolom_categorical', f'distribusi_{col}.png')

    plt.savefig(file_path)
    st.pyplot(plt)
    plt.close()  # Tutup plot untuk menghemat memori


# In[59]:


#  CEK MISSING VALUES, UNIQUE VALUES, DATA DUPLIKAT, DAN OUTLIERS
st.subheader("CEK KUALITAS DATA")


# In[60]:


# 4.1 Missing Values
st.write("\nMISSING VALUES:")
missing_values = df.isnull().sum()
missing_percent = (missing_values / len(df)) * 100
missing_data = pd.DataFrame({'Missing Values': missing_values,
                             'Percentage (%)': missing_percent})
st.dataframe(missing_data[missing_data['Missing Values'] > 0])


# In[61]:


# Visualisasi missing values
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
plt.title('Missing Values Heatmap')
plt.tight_layout()
file_path = os.path.join('Assets/missing_values', 'missing_values.png')
plt.savefig(file_path)
st.pyplot(plt)
plt.close()  # Tutup plot untuk menghemat memori


# In[62]:


#  Unique Values
st.write("\nUNIQUE VALUES:")
unique_values = pd.DataFrame({
    'Column': df.columns,
    'Data Type': df.dtypes,
    'Unique Values': [df[col].nunique() for col in df.columns],
    'Unique Ratio (%)': [(df[col].nunique() / len(df)) * 100 for col in df.columns]
})
st.dataframe(unique_values)


# In[63]:


# Data Duplikat
st.write("\nDATA DUPLIKAT:")
duplicates = df.duplicated().sum()
st.write(f"Jumlah baris duplikat: {duplicates} ({(duplicates/len(df))*100:.2f}%)")


# In[64]:


# Keseimbangan Data (untuk data kategorikal dan target )
categorical_cols = df.select_dtypes(include=['object']).columns
if len(categorical_cols) > 0:
    st.write("\nKESEIMBANGAN DATA KATEGORIKAL:")
    for col in categorical_cols:
        if df[col].nunique() < 10:  # Hanya tampilkan jika jumlah kelas < 10
            class_counts = df[col].value_counts()
            class_percents = (class_counts / len(df)) * 100

            st.write(f"\nDistribusi untuk {col}:")
            for cls, count in class_counts.items():
                st.write(f"  {cls}: {count} ({class_percents[cls]:.2f}%)")

            # Visualisasi keseimbangan
            plt.figure(figsize=(10, 6))
            sns.countplot(y=df[col])
            plt.title(f'Distribusi Kelas untuk {col}')
            plt.tight_layout()
            file_path = os.path.join('Assets/class_balance/categorical', f'class_balance_{col}.png')
            plt.savefig(file_path)
            st.pyplot(plt)
            plt.close()


# In[65]:


# DETEKSI OUTLIER
st.subheader("DETEKSI OUTLIER (MENGGUNAKAN METODE IQR):")


# In[66]:


# Ganti '?' dengan NaN
df = df.replace('?', np.nan)


# In[67]:


# Inisialisasi dictionary untuk menyimpan informasi outlier
outlier_summary = {}


# In[68]:


# Ringkasan outlier untuk kolom numerik dan kategorikal
# Untuk kolom numerik
for col in kolom_numerik:
    if df[col].isna().all():
        st.write(f"Kolom {col}: Tidak dapat menghitung outlier (semua data NaN)")
        continue

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
    outlier_count = len(outliers)
    outlier_summary[col] = outlier_count

# Untuk kolom kategorikal
st.write("\nFrekuensi tidak umum pada kolom kategorikal:")
for col in df.select_dtypes(include=['object']).columns:
    value_counts = df[col].value_counts()
    total_count = len(df)
    rare_values = value_counts[value_counts/total_count < 0.01]  # kurang dari 1%

    if len(rare_values) > 0:
        st.write(f"\n{col}:")
        for value, count in rare_values.items():
            st.write(f"  {value}: {count} ({(count/total_count)*100:.2f}%)")
st.write("\nRingkasan Outlier:")
if all(count == 0 for count in outlier_summary.values()):
    st.write("  Tidak ada outlier yang ditemukan di semua kolom numerik.")
else:
    for col, count in outlier_summary.items():
        if count > 0:
            st.write(f"  {col}: {count} outlier ditemukan")


# In[69]:


# 5. KESIMPULAN
st.subheader("KESIMPULAN EDA")
st.write("\nBerikut kesimpulan dari proses EDA:")
st.write("1. Dataset memiliki {} baris dan {} kolom.".format(df.shape[0], df.shape[1]))


# In[70]:


# Missing values
if missing_values.sum() > 0:
    st.write("2. Terdapat {} missing values pada {} kolom.".format(
        missing_values.sum(), len(missing_data[missing_data['Missing Values'] > 0])))
else:
    st.write("2. Tidak terdapat missing values pada dataset.")


# In[71]:


# Duplicates
if duplicates > 0:
    st.write("3. Terdapat {} baris duplikat ({:.2f}%).".format(duplicates, (duplicates/len(df))*100))
else:
    st.write("3. Tidak terdapat data duplikat pada dataset.")


# In[72]:


# Outliers
# Initialize empty list for outlier count
outlier_count = []

# Check outliers in numeric columns using IQR method
for col in kolom_numerik:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)][col]
    if len(outliers) > 0:
        outlier_count.append(col)

# Check rare values in categorical columns (values appearing less than 1% of the time)
for col in df.select_dtypes(include=['object']).columns:
    value_counts = df[col].value_counts()
    total_count = len(df)
    rare_values = value_counts[value_counts/total_count < 0.01]  # less than 1%
    if len(rare_values) > 0:
        outlier_count.append(col)

if len(outlier_count) > 0:
    st.write("4. Terdeteksi outlier pada kolom: {}.".format(", ".join(outlier_count)))
else:
    st.write("4. Tidak terdeteksi outlier pada dataset.")


# In[73]:


# 2. Tpreprocessing data
st.subheader("2. PREPROCESSING DATA")



# In[74]:


# tangani missing values di setiap kolom
st.write("=== Missing Values ===")
st.write("Missing values sebelum imputasi:")
st.write(df.isnull().sum())

# Imputasi missing values untuk kolom numerik dengan median
for col in kolom_numerik:
    if df[col].isnull().any():
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)

# Imputasi missing values untuk kolom kategorikal dengan modus
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if df[col].isnull().any():
        mode_value = df[col].mode()[0]
        df[col] = df[col].fillna(mode_value)

st.write("\nMissing values setelah imputasi:")
st.dataframe(df.isnull().sum())


# In[75]:


# Perbaikan ketidakkonsistensinan data untuk kolom numerik
st.write("=== Perbaikan Ketidakkonsistenan Data Numerik ===")

# Age harus berupa bilangan bulat
df['Age'] = df['Age'].round().astype(int)

# Height dan Weight harus positif dan dalam range yang masuk akal
df['Height'] = df['Height'].clip(lower=0, upper=300)  # dalam cm
df['Weight'] = df['Weight'].round().clip(lower=0, upper=500).astype(int)  # dalam kg

# FCVC (Frequency of consumption of vegetables) range 1-3
df['FCVC'] = df['FCVC'].clip(lower=1, upper=3)

# NCP (Number of main meals) range 1-4
df['NCP'] = df['NCP'].clip(lower=1, upper=4)

# CH2O (Consumption of water) range 1-3
df['CH2O'] = df['CH2O'].clip(lower=1, upper=3)

# FAF (Physical activity frequency) range 0-3
df['FAF'] = df['FAF'].clip(lower=0, upper=3)

# TUE (Time using technology devices) range 0-2
df['TUE'] = df['TUE'].clip(lower=0, upper=2)

st.write("\nRange nilai setelah perbaikan:")
st.dataframe(df[kolom_numerik].head(25))


# In[76]:


# Perbaikan ketidakkonsistensinan data untuk kolom kategorikal
st.write("Memperbaiki konsistensi data kategorikal...")

# Mengubah semua nilai kategorikal menjadi lowercase
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = df[col].str.lower()

st.write("\nSample data setelah perbaikan:")
st.dataframe(df[categorical_cols].head(10))


# In[77]:


# Show number of duplicates before removal
st.write("Jumlah data duplikat sebelum penanganan:", df.duplicated().sum())

# Remove duplicate rows while keeping the first occurrence
df = df.drop_duplicates(keep='first')

# Show number of duplicates after removal
st.write("Jumlah data duplikat setelah penanganan:", df.duplicated().sum())

# Reset index after removing duplicates
df = df.reset_index(drop=True)

st.write("\nUkuran dataset setelah penanganan duplikasi:")
st.dataframe(df.shape)


# In[78]:


st.write("Mengubah kolom kategorikal menjadi numerik...")
# Bikin salinan dataframe biar data asli gak rusak
df_encoded = df.copy()

# Encoding untuk kolom yang cuma punya 2 kategori (binary)
# Contoh: Gender (Male/Female), FAVC (yes/no), SMOKE (yes/no)
binary_cols = ['Gender', 'FAVC', 'SCC', 'SMOKE', 'family_history_with_overweight']

for col in binary_cols:
    le = LabelEncoder()  # Bikin encoder baru untuk setiap kolom
    df_encoded[col] = le.fit_transform(df_encoded[col])
    # Hasil: Male→0, Female→1 atau yes→1, no→0

# One-hot encoding untuk kolom yang punya lebih dari 2 kategori
# Contoh: MTRANS (car, walking, bike, etc) jadi beberapa kolom terpisah
non_binary_cols = ['CALC', 'CAEC', 'MTRANS']
df_encoded = pd.get_dummies(df_encoded, columns=non_binary_cols, 
                           prefix=non_binary_cols, dtype=int)
# dtype=int biar hasilnya 0/1, bukan True/False

# Encode target variable (yang mau diprediksi) pake encoder terpisah
target_encoder = LabelEncoder()
df_encoded['NObeyesdad'] = target_encoder.fit_transform(df_encoded['NObeyesdad'])
# Hasil: tingkat obesitas jadi angka 0,1,2,3,dst

# Tampilkan hasil encoding
st.write("Shape setelah encoding:", df_encoded.shape)
st.write("\nBeberapa baris pertama data yang sudah di-encode:")
st.dataframe(df_encoded.head())

# Cek tipe data untuk memastikan semuanya numerik
st.write("\nTipe data setelah encoding:")
st.dataframe(df_encoded.dtypes)

# Lihat mapping target variable biar paham konversinya
st.write("\nMapping target variable:")
st.write(f"Nilai asli: {df['NObeyesdad'].unique()}")
st.write(f"Nilai setelah encode: {df_encoded['NObeyesdad'].unique()}")


# In[79]:


# menampilkan kolom target setelah encoding
st.write("\nEncoded NObeyesdad values:")
st.dataframe(df['NObeyesdad'].head(10))

st.write("\nEncoded NObeyesdad values:")
st.dataframe(df_encoded['NObeyesdad'].head(10))



# In[80]:


st.write("\nMenampilkan kolom yang paling berpengaruh terhadap kolom target")
X = df_encoded.drop('NObeyesdad', axis=1)  # Fitur
y = df['NObeyesdad']  # Target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Latih model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Cek feature importance
importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
st.dataframe(importance)


# In[81]:


# pemilihan fitur menggunakan RFE (Recursive Feature Elimination)

# Get the top 10 features based on importance ranking
top_10_features = importance.nlargest(10).index.tolist()
st.write("Top 10 features:", top_10_features)

# Create a new RandomForestClassifier
selector = RandomForestClassifier(random_state=42)

# Create RFE with n_features_to_select=10
rfe = RFE(estimator=selector, n_features_to_select=10, step=1)

# Fit RFE
X_selected = rfe.fit_transform(X, y)

# Create new dataframe with selected features
X_selected_df = pd.DataFrame(X_selected, columns=np.array(X.columns)[rfe.support_])

st.write("\nSelected features:")
for i, feature in enumerate(X_selected_df.columns, 1):
    st.write(f"{i}. {feature}")


# In[82]:


# Peneyimbangan data dengan SMOTE
selected_features = top_10_features + ['NObeyesdad']
X_selected = df_encoded[top_10_features]
y = df_encoded['NObeyesdad']

# Initialize and apply SMOTE
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_selected, y)

# Create figure with two subplots
plt.figure(figsize=(15, 5))

# Plot before SMOTE
plt.subplot(1, 2, 1)
before_counts = y.value_counts()
plt.bar(range(len(before_counts)), before_counts.values)
plt.title('Class Distribution Before SMOTE')
plt.xticks(range(len(before_counts)), before_counts.index, rotation=45)
plt.xlabel('NObeyesdad')
plt.ylabel('Count')

# Plot after SMOTE
plt.subplot(1, 2, 2)
after_counts = pd.Series(y_balanced).value_counts()
plt.bar(range(len(after_counts)), after_counts.values)
plt.title('Class Distribution After SMOTE')
plt.xticks(range(len(after_counts)), after_counts.index, rotation=45)
plt.xlabel('NObeyesdad')
plt.ylabel('Count')

plt.tight_layout()
st.pyplot(plt)
plt.close()

# Print class distribution before and after SMOTE
st.write("\nClass distribution before SMOTE:")
st.dataframe(pd.Series(y).value_counts())
st.write("\nClass distribution after SMOTE:")
st.dataframe(pd.Series(y_balanced).value_counts())

# ===== VISUALISASI TOP 10 FITUR DENGAN BAR CHART =====

# Gabungkan fitur dan target untuk perbandingan
df_before = pd.DataFrame(X_selected, columns=top_10_features)
df_after = pd.DataFrame(X_balanced, columns=top_10_features)

# Buat statistik ringkasan untuk setiap fitur
fig, axes = plt.subplots(2, 5, figsize=(20, 10))
fig.suptitle('Perbandingan Rata-rata Top 10 Fitur: Sebelum vs Sesudah SMOTE', fontsize=16, fontweight='bold')

for i, feature in enumerate(top_10_features):
    row = i // 5
    col = i % 5

    # Hitung rata-rata sebelum dan sesudah SMOTE
    mean_before = df_before[feature].mean()
    mean_after = df_after[feature].mean()

    # Buat bar chart
    categories = ['Sebelum SMOTE', 'Sesudah SMOTE']
    values = [mean_before, mean_after]
    colors = ['red', 'blue']

    bars = axes[row, col].bar(categories, values, color=colors, alpha=0.7)
    axes[row, col].set_title(f'{feature}', fontweight='bold')
    axes[row, col].set_ylabel('Rata-rata')
    axes[row, col].grid(True, alpha=0.3)

    # Tambahkan nilai di atas bar
    for bar, value in zip(bars, values):
        axes[row, col].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.2f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
st.pyplot(plt)
plt.close()

st.write('''
      Seperti yang terlihat, nilai rata-rata fitur seperti berat badan turun sedikit dari 88,80 menjadi 87,18 kg, 
      sementara tinggi badan dan jumlah makan utama tetap stabil. Perubahan ini sangat kecil, 
      menunjukkan SMOTE berhasil menyeimbangkan data tanpa mengubah karakteristik utama. 
      Ini penting untuk memastikan model machine learning kita bekerja lebih baik dengan data yang seimbang''')

# Tampilkan perbandingan jumlah data
st.write("\nPERBANDINGAN JUMLAH DATA:")

fig2, ax = plt.subplots(1, 1, figsize=(8, 6))
categories = ['Sebelum SMOTE', 'Sesudah SMOTE']  
values = [len(df_before), len(df_after)]
colors = ['red', 'blue']

bars = ax.bar(categories, values, color=colors, alpha=0.7)
ax.set_title('Perbandingan Jumlah Data', fontsize=14, fontweight='bold')
ax.set_ylabel('Jumlah Data')
ax.grid(True, alpha=0.3)

# Tambahkan nilai di atas bar
for bar, value in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
           f'{value}', ha='center', va='bottom', fontweight='bold', fontsize=12)

st.pyplot(plt)
plt.close()

# Tampilkan ringkasan sederhana
st.write(f"\nRINGKASAN BALANCING:")
st.write(f"Data sebelum SMOTE: {X_selected.shape[0]} baris")
st.write(f"Data sesudah SMOTE: {X_balanced.shape[0]} baris")
st.write(f"Penambahan data: {X_balanced.shape[0] - X_selected.shape[0]} baris")
st.write(f"Fitur yang digunakan: {len(top_10_features)} fitur")
st.write(f"Semua kelas target sekarang seimbang: {len(set(y_balanced))} kelas dengan masing-masing {max(pd.Series(y_balanced).value_counts())} data")


# In[83]:


# normaliassi atau standarisasi data
# Buat objek StandardScaler
scaler = StandardScaler()

# Lakukan standarisasi pada fitur-fitur numerik X_balanced
X_balanced_scaled = X_balanced.copy()
numeric_columns = X_balanced.select_dtypes(include=['float64', 'int64']).columns
X_balanced_scaled[numeric_columns] = scaler.fit_transform(X_balanced[numeric_columns])

st.write("\nPerbandingan statistik sebelum dan sesudah standarisasi:")
st.write("\nSebelum standarisasi:")
st.dataframe(X_balanced[numeric_columns].describe())
st.write("\nSetelah standarisasi:")
st.dataframe(X_balanced_scaled[numeric_columns].describe())

# Visualisasi distribusi data sebelum dan sesudah standarisasi
plt.figure(figsize=(15, 5))

# Plot sebelum standarisasi
plt.subplot(1, 2, 1)
plt.boxplot(X_balanced[numeric_columns])
plt.title('Distribusi Sebelum Standarisasi')
plt.xticks(range(1, len(numeric_columns) + 1), numeric_columns, rotation=45)

# Plot setelah standarisasi
plt.subplot(1, 2, 2)
plt.boxplot(X_balanced_scaled[numeric_columns])
plt.title('Distribusi Setelah Standarisasi')
plt.xticks(range(1, len(numeric_columns) + 1), numeric_columns, rotation=45)

plt.tight_layout()
st.pyplot(plt)
plt.close()


# In[84]:


# Menyimpan top 10 feature dan kolom target yang dipilih ke daalam file csv
# Create dataframe with top 10 features and target column
selected_columns = top_10_features + ['NObeyesdad']
df_selected = df[selected_columns].copy()

# Save to CSV file
df_selected.to_csv('selected_features_data.csv', index=False)

st.write("Data has been saved to 'selected_features_data.csv'")
st.write(f"Saved {len(selected_columns)} columns: {', '.join(selected_columns)}")
st.write(f"Total rows: {len(df_selected)}")


# In[85]:


# 3. Pemodelan dan Evaluasi

st.subheader("3. Pemodelan dan Evaluasi")



# In[86]:


warnings.filterwarnings('ignore')

X = X_balanced_scaled
y = y_balanced

# Encode categorical variables
categorical_cols = X.select_dtypes(include=['object']).columns
for col in categorical_cols:
    X[col] = pd.get_dummies(X[col], drop_first=True)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}

# Train and evaluate models
results = {}

for name, model in models.items():
    st.write(f"\nTraining {name}...")

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Store results
    results[name] = {
        'accuracy': accuracy,
        'predictions': y_pred
    }

    # Print results
    st.write(f"{name} Accuracy: {accuracy:.4f}")
    st.write("\nClassification Report:")
    st.write(classification_report(y_test, y_pred))

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    st.pyplot(plt)
    plt.close()

# Compare model performances
accuracies = {name: result['accuracy'] for name, result in results.items()}
plt.figure(figsize=(10, 6))
plt.bar(accuracies.keys(), accuracies.values())
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
for i, v in enumerate(accuracies.values()):
    plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
plt.tight_layout()
st.pyplot(plt)
plt.close()

# Print final comparison
st.write("\nFinal Model Comparison:")
for name, accuracy in accuracies.items():
    st.write(f"{name}: {accuracy:.4f}")


# Berdasarkan hasil evaluasi ketiga model machine learning untuk prediksi tingkat obesitas, **Random Forest menunjukkan performa yang sangat superior** dengan akurasi 93.54%, jauh melampaui Logistic Regression (70.33%) dan Decision Tree yang belum selesai dijalankan. Random Forest menunjukkan keunggulan dalam semua metrik evaluasi, dengan precision, recall, dan f1-score yang konsisten tinggi di atas 0.86 untuk semua kategori obesitas. Khususnya untuk kategori "obesity_type_iii", model Random Forest mencapai precision sempurna (1.00) dan recall 0.99, menunjukkan kemampuan luar biasa dalam mengidentifikasi kasus obesitas tingkat III.
# 
# Sebaliknya, **Logistic Regression menunjukkan performa yang kurang memuaskan** dengan akurasi hanya 70.33% dan kesulitan dalam mengklasifikasikan beberapa kategori seperti "normal_weight" (recall 0.36) dan "overweight_level_i" (recall 0.32). Perbedaan performa yang signifikan ini mengindikasikan bahwa hubungan antar fitur dalam dataset obesitas bersifat non-linear dan kompleks, sehingga algoritma ensemble seperti Random Forest yang mampu menangkap pola kompleks melalui kombinasi multiple decision trees lebih efektif dibandingkan model linear seperti Logistic Regression. **Rekomendasi untuk implementasi praktis adalah menggunakan Random Forest** sebagai model utama untuk sistem prediksi tingkat obesitas.

# In[87]:


# 4. Hyperparameter Tuning

st.subheader("4. Hyperparameter Tuning")



# In[88]:


# Import library tambahan untuk hyperparameter tuning
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score
import numpy as np
import time

st.write("HYPERPARAMETER TUNING DAN OPTIMASI MODEL")


# Definisikan grid parameter untuk setiap model
param_grids = {
    'Random Forest': {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    },
    'Logistic Regression': {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l2'],  # Hapus 'l1' dulu
    'solver': ['lbfgs'],  # Ganti ke solver yang lebih stabil
    'max_iter': [5000, 10000]  # Naikin lebih tinggi
    },
    'Decision Tree': {
        'max_depth': [None, 5, 10, 15, 20, 25],
        'min_samples_split': [2, 5, 10, 15, 20],
        'min_samples_leaf': [1, 2, 4, 6, 8],
        'criterion': ['gini', 'entropy'],
        'max_features': ['sqrt', 'log2', None]
    }
}

# Inisialisasi dictionary untuk model yang sudah dioptimasi
optimized_models = {}
optimization_results = {}

# Lakukan hyperparameter tuning untuk setiap model
for name, model in models.items():
    st.write(f"\n{'-'*40}")
    st.write(f"Mengoptimalkan {name}...")
    st.write(f"{'-'*40}")

    start_time = time.time()

    # Gunakan RandomizedSearchCV untuk komputasi lebih cepat (terutama untuk Random Forest)
    if name == 'Random Forest':
        search = RandomizedSearchCV(
            model, 
            param_grids[name], 
            n_iter=50,  # Jumlah pengaturan parameter yang diambil sampelnya
            cv=5, 
            scoring='accuracy',
            n_jobs=-1, 
            random_state=42,
            verbose=1
        )
    else:
        # Gunakan GridSearchCV untuk Logistic Regression dan Decision Tree
        search = GridSearchCV(
            model, 
            param_grids[name], 
            cv=5, 
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )

    # Fit pencarian
    search.fit(X_train, y_train)

    end_time = time.time()

    # Simpan model terbaik
    optimized_models[name] = search.best_estimator_

    # Simpan hasil optimasi
    optimization_results[name] = {
        'best_params': search.best_params_,
        'best_score': search.best_score_,
        'optimization_time': end_time - start_time
    }

    st.write(f"Parameter terbaik untuk {name}:")
    for param, value in search.best_params_.items():
        st.write(f"  {param}: {value}")
    st.write(f"Skor cross-validation terbaik: {search.best_score_:.4f}")
    st.write(f"Waktu optimasi: {end_time - start_time:.2f} detik")

st.write(f"\n{'='*60}")
st.write("EVALUASI MODEL YANG SUDAH DIOPTIMASI")
st.write(f"{'='*60}")

# Evaluasi model yang sudah dioptimasi
optimized_results = {}

for name, model in optimized_models.items():
    st.write(f"\nMengevaluasi {name} yang Sudah Dioptimasi...")

    # Buat prediksi
    y_pred = model.predict(X_test)

    # Hitung metrik
    accuracy = accuracy_score(y_test, y_pred)

    # Simpan hasil
    optimized_results[name] = {
        'accuracy': accuracy,
        'predictions': y_pred
    }

    st.write(f"Akurasi {name} yang Sudah Dioptimasi: {accuracy:.4f}")
    st.write(f"Peningkatan: {accuracy - results[name]['accuracy']:.4f}")
    st.write("\nLaporan Klasifikasi:")
    st.write(classification_report(y_test, y_pred))

    # Plot confusion matrix untuk model yang sudah dioptimasi
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
    plt.title(f'Confusion Matrix - {name} yang Sudah Dioptimasi')
    plt.ylabel('Label Sebenarnya')
    plt.xlabel('Label Prediksi')
    plt.tight_layout()
    st.pyplot(plt)
    plt.close()


st.write("PERBANDINGAN PERFORMA SEBELUM DAN SESUDAH OPTIMASI")


# Buat visualisasi perbandingan
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Bar chart perbandingan
model_names = list(results.keys())
original_accuracies = [results[name]['accuracy'] for name in model_names]
optimized_accuracies = [optimized_results[name]['accuracy'] for name in model_names]

x = np.arange(len(model_names))
width = 0.35

axes[0, 0].bar(x - width/2, original_accuracies, width, label='Asli', alpha=0.8, color='skyblue')
axes[0, 0].bar(x + width/2, optimized_accuracies, width, label='Dioptimasi', alpha=0.8, color='lightgreen')

axes[0, 0].set_xlabel('Model')
axes[0, 0].set_ylabel('Akurasi')
axes[0, 0].set_title('Akurasi Model: Asli vs Dioptimasi')
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(model_names, rotation=45)
axes[0, 0].legend()
axes[0, 0].set_ylim(0, 1)

# Tambahkan label nilai pada bar
for i, (orig, opt) in enumerate(zip(original_accuracies, optimized_accuracies)):
    axes[0, 0].text(i - width/2, orig + 0.01, f'{orig:.3f}', ha='center', va='bottom')
    axes[0, 0].text(i + width/2, opt + 0.01, f'{opt:.3f}', ha='center', va='bottom')

# 2. Chart peningkatan
improvements = [optimized_results[name]['accuracy'] - results[name]['accuracy'] for name in model_names]
colors = ['green' if imp >= 0 else 'red' for imp in improvements]

axes[0, 1].bar(model_names, improvements, color=colors, alpha=0.7)
axes[0, 1].set_xlabel('Model')
axes[0, 1].set_ylabel('Peningkatan Akurasi')
axes[0, 1].set_title('Peningkatan Akurasi Setelah Optimasi')
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)

# Tambahkan label nilai
for i, imp in enumerate(improvements):
    axes[0, 1].text(i, imp + 0.001 if imp >= 0 else imp - 0.001, 
                   f'{imp:.4f}', ha='center', va='bottom' if imp >= 0 else 'top')

# 3. Chart waktu optimasi
opt_times = [optimization_results[name]['optimization_time'] for name in model_names]
axes[1, 0].bar(model_names, opt_times, color='orange', alpha=0.7)
axes[1, 0].set_xlabel('Model')
axes[1, 0].set_ylabel('Waktu Optimasi (detik)')
axes[1, 0].set_title('Waktu Hyperparameter Tuning')
axes[1, 0].tick_params(axis='x', rotation=45)

# Tambahkan label nilai
for i, time_val in enumerate(opt_times):
    axes[1, 0].text(i, time_val + max(opt_times)*0.02, f'{time_val:.1f}s', ha='center', va='bottom')

# 4. Skor CV terbaik
best_cv_scores = [optimization_results[name]['best_score'] for name in model_names]
axes[1, 1].bar(model_names, best_cv_scores, color='purple', alpha=0.7)
axes[1, 1].set_xlabel('Model')
axes[1, 1].set_ylabel('Skor CV Terbaik')
axes[1, 1].set_title('Skor Cross-Validation Terbaik')
axes[1, 1].tick_params(axis='x', rotation=45)
axes[1, 1].set_ylim(0, 1)

# Tambahkan label nilai
for i, score in enumerate(best_cv_scores):
    axes[1, 1].text(i, score + 0.01, f'{score:.3f}', ha='center', va='bottom')

plt.tight_layout()
st.pyplot(plt)
plt.close()

# st.dataframe tabel perbandingan detail
st.write("\nTABEL PERBANDINGAN DETAIL:")
st.write("="*80)
st.write(f"{'Model':<20} {'Asli':<12} {'Dioptimasi':<12} {'Peningkatan':<12} {'Skor CV':<12}")
st.write("-"*80)

for name in model_names:
    original_acc = results[name]['accuracy']
    optimized_acc = optimized_results[name]['accuracy']
    improvement = optimized_acc - original_acc
    cv_score = optimization_results[name]['best_score']

    st.write(f"{name:<20} {original_acc:<12.4f} {optimized_acc:<12.4f} {improvement:<12.4f} {cv_score:<12.4f}")

st.write("-"*80)

# Cari model dengan performa terbaik
best_model_name = max(optimized_results.keys(), key=lambda x: optimized_results[x]['accuracy'])
best_accuracy = optimized_results[best_model_name]['accuracy']

st.write(f"\nMODEL DENGAN PERFORMA TERBAIK: {best_model_name}")
st.write(f"Akurasi: {best_accuracy:.4f}")
st.write(f"Parameter Terbaik: {optimization_results[best_model_name]['best_params']}")


