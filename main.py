import pandas as pd

# MELAKUKAN EDA DATASET

# Load the dataset
file_path = "ObesityDataSet.csv"
df = pd.read_csv(file_path)

# 1. Tampilkan beberapa baris pertama dan informasi umum dataset
print(df.head(5))
print(df.info())
print(df.describe())
print("Jumlah baris dan kolom:", df.shape)
