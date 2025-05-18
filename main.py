import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

# 1. LOAD DATA
df = pd.read_csv('/content/drive/MyDrive/BENGKEL KODING/ObesityDataSet.csv')

# 2. TAMPILKAN BEBERAPA BARIS PERTAMA DAN INFORMASI UMUM DATASET
print("="*50)
print("EXPLORATORY DATA ANALYSIS (EDA)")
print("="*50)

# Menampilkan 5 baris pertama
print("\n2.1 DATA PREVIEW (5 BARIS PERTAMA):")
print(df.head())

# Informasi umum dataset
print("\n2.2 INFORMASI UMUM DATASET:")
print(f"Jumlah baris: {df.shape[0]}")
print(f"Jumlah kolom: {df.shape[1]}")
print(f"Nama kolom: {', '.join(df.columns)}")

# Deskripsi dan tipe data
print("\n2.3 TIPE DATA:")
print(df.dtypes)

print("\n2.4 DESKRIPSI STATISTIK DATA NUMERIK:")
print(df.describe())

# Untuk kolom kategorikal
categorical_cols = df.select_dtypes(include=['object']).columns
if len(categorical_cols) > 0:
    print("\n2.5 DESKRIPSI DATA KATEGORIKAL:")
    for col in categorical_cols:
        print(f"\nKolom: {col}")
        print(df[col].value_counts())
        print(f"Jumlah nilai unik: {df[col].nunique()}")

# 3. VISUALISASI DATA
print("="*50)
print("VISUALISASI DATA")
print("="*50)

# Pengaturan untuk tampilan grafik
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (14, 8)

# 3.1 Distribusi variabel kategorikal
if len(categorical_cols) > 0:
    print("\n3.2 DISTRIBUSI VARIABEL KATEGORIKAL")
    
    # Hanya tampilkan 6 variabel kategorikal pertama (jika ada)
    for i, col in enumerate(categorical_cols[:min(6, len(categorical_cols))]):
        plt.figure(figsize=(12, 6))
        top_categories = df[col].value_counts().head(10)  # Ambil 10 kategori teratas
        sns.barplot(x=top_categories.index, y=top_categories.values)
        plt.title(f'Distribusi {col} (10 Kategori Teratas)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'distribusi_{col}.png')
        plt.show()


# 4. CEK MISSING VALUES, UNIQUE VALUES, DATA DUPLIKAT, DAN OUTLIERS
print("="*50)
print("CEK KUALITAS DATA")
print("="*50)

# 4.1 Missing Values
print("\n4.1 MISSING VALUES:")
missing_values = df.isnull().sum()
missing_percent = (missing_values / len(df)) * 100
missing_data = pd.DataFrame({'Missing Values': missing_values, 
                             'Percentage (%)': missing_percent})
print(missing_data[missing_data['Missing Values'] > 0])

# Visualisasi missing values
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
plt.title('Missing Values Heatmap')
plt.tight_layout()
plt.savefig('missing_values.png')
plt.show()

# 4.2 Unique Values
print("\n4.2 UNIQUE VALUES:")
unique_values = pd.DataFrame({
    'Column': df.columns,
    'Data Type': df.dtypes,
    'Unique Values': [df[col].nunique() for col in df.columns],
    'Unique Ratio (%)': [(df[col].nunique() / len(df)) * 100 for col in df.columns]
})
print(unique_values)

# 4.3 Data Duplikat
print("\n4.3 DATA DUPLIKAT:")
duplicates = df.duplicated().sum()
print(f"Jumlah baris duplikat: {duplicates} ({(duplicates/len(df))*100:.2f}%)")

# 4.4 Keseimbangan Data (untuk data kategorikal yang bisa menjadi target)
if len(categorical_cols) > 0:
    print("\n4.4 KESEIMBANGAN DATA KATEGORIKAL:")
    for col in categorical_cols:
        if df[col].nunique() < 10:  # Hanya tampilkan jika jumlah kelas < 10
            class_counts = df[col].value_counts()
            class_percents = (class_counts / len(df)) * 100
            
            print(f"\nDistribusi untuk {col}:")
            for cls, count in class_counts.items():
                print(f"  {cls}: {count} ({class_percents[cls]:.2f}%)")
            
            # Visualisasi keseimbangan
            plt.figure(figsize=(10, 6))
            sns.countplot(y=df[col])
            plt.title(f'Distribusi Kelas untuk {col}')
            plt.tight_layout()
            plt.savefig(f'class_balance_{col}.png')
            plt.show()

# 4.5 DETEKSI NILAI TIDAK UMUM PADA DATA KATEGORIKAL
# print("\n4.5 DETEKSI NILAI TIDAK UMUM PADA DATA KATEGORIKAL:")

# # Pastikan categorical_cols sudah didefinisikan, jika belum:
# # categorical_cols = df.select_dtypes(include=['object', 'category']).columns

# for col in categorical_cols:
#     print(f"\nKolom: {col}")
    
#     # Hitung frekuensi nilai
#     value_counts = df[col].value_counts()
#     value_percent = (value_counts / len(df)) * 100
    
#     # Tentukan nilai yang jarang muncul (misalnya kurang dari 5%)
#     rare_threshold = 5.0
#     rare_values = value_counts[value_percent < rare_threshold]
#     rare_count = rare_values.sum()
#     rare_percent = (rare_count / len(df)) * 100
    
#     print(f"  Jumlah nilai unik: {df[col].nunique()}")
#     print(f"  Nilai yang jarang muncul (<{rare_threshold}%): {len(rare_values)} kategori")
#     print(f"  Total data dengan nilai jarang: {rare_count} ({rare_percent:.2f}%)")
    
#     # Tampilkan 5 nilai teratas (yang paling sering muncul)
#     print(f"  5 nilai teratas:")
#     for i, (val, count) in enumerate(value_counts.head(5).items()):
#         percent = (count / len(df)) * 100
#         print(f"    {i+1}. {val}: {count} ({percent:.2f}%)")
    
#     # Tampilkan beberapa nilai yang jarang muncul (jika ada)
#     if len(rare_values) > 0:
#         print(f"  Contoh nilai yang jarang muncul:")
#         for i, (val, count) in enumerate(rare_values.head(5).items()):
#             percent = (count / len(df)) * 100
#             print(f"    - {val}: {count} ({percent:.2f}%)")
    
#     # Visualisasi distribusi nilai (hanya untuk kolom dengan nilai unik < 20)
#     if df[col].nunique() < 20:
#         plt.figure(figsize=(12, 6))
        
#         # Bar plot untuk menunjukkan frekuensi setiap nilai
#         sns.countplot(y=df[col], order=value_counts.index)
#         plt.title(f'Distribusi Nilai untuk {col}')
#         plt.xlabel('Jumlah')
#         plt.ylabel(col)
#         plt.tight_layout()
#         plt.show()
#     else:
#         # Untuk kategori banyak, tampilkan 10 teratas dan sisanya sebagai "Lainnya"
#         plt.figure(figsize=(12, 6))
        
#         # Ambil 10 nilai teratas
#         top_values = value_counts.head(10)
#         # Gabungkan sisanya sebagai "Lainnya"
#         other_sum = value_counts[10:].sum()
        
#         # Buat series baru dengan 10 teratas + "Lainnya"
#         plot_data = pd.Series(top_values.values, index=top_values.index)
#         if other_sum > 0:
#             plot_data["Lainnya"] = other_sum
        
#         # Plot
#         plt.pie(plot_data, labels=plot_data.index, autopct='%1.1f%%')
#         plt.title(f'Distribusi 10 Nilai Teratas untuk {col}')
#         plt.axis('equal')
#         plt.tight_layout()
#         plt.show()

# 5. KESIMPULAN
print("="*50)
print("KESIMPULAN EDA")
print("="*50)
print("\nBerikut kesimpulan dari proses EDA:")
print("1. Dataset memiliki {} baris dan {} kolom.".format(df.shape[0], df.shape[1]))

# Missing values
if missing_values.sum() > 0:
    print("2. Terdapat {} missing values pada {} kolom.".format(
        missing_values.sum(), len(missing_data[missing_data['Missing Values'] > 0])))
else:
    print("2. Tidak terdapat missing values pada dataset.")

# Duplicates
if duplicates > 0:
    print("3. Terdapat {} baris duplikat ({:.2f}%).".format(duplicates, (duplicates/len(df))*100))
else:
    print("3. Tidak terdapat data duplikat pada dataset.")

# Outliers
outlier_cols = []
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)][col]
    if len(outliers) > 0:
        outlier_cols.append(col)

if len(outlier_cols) > 0:
    print("4. Terdeteksi outlier pada kolom: {}.".format(", ".join(outlier_cols)))
else:
    print("4. Tidak terdeteksi outlier pada dataset.")