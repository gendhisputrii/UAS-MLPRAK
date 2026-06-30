import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv1D, MaxPooling1D, Flatten, Dense,
                                      Dropout, LSTM)
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# 1. INPUT
print("=" * 60)
print("1. INPUT")
print("=" * 60)

df = pd.read_csv("kidney_disease.csv")
print(f"Shape awal data: {df.shape}")
print(df.head())


# 2. PREPROCESSING
print("\n" + "=" * 60)
print("2. PREPROCESSING")
print("=" * 60)

df = df.drop(columns=["id"])

for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype(str).str.strip().str.replace("\t", "", regex=False)
    df[col] = df[col].replace({"nan": np.nan, "?": np.nan, "": np.nan})

df["classification"] = df["classification"].replace({"ckd": "ckd", "notckd": "notckd"})
print("\nDistribusi target setelah dibersihkan:")
print(df["classification"].value_counts())

categorical_cols = ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"]
numerical_cols = [c for c in df.columns if c not in categorical_cols + ["classification"]]

for col in numerical_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Cek Missing Value 
print("\n--- Cek Missing Value per Kolom (sebelum imputasi) ---")
print(df.isnull().sum())

# Handling Missing Value 
for col in numerical_cols:
    df[col] = df[col].fillna(df[col].median())
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print(f"\nTotal missing value setelah imputasi: {df.isnull().sum().sum()}")

# Deteksi Outlier (Z-score) 
z_scores = np.abs(stats.zscore(df[numerical_cols]))
outliers_z = df[(z_scores > 3).any(axis=1)]
print(f"\n--- Deteksi Outlier (Z-score > 3) ---")
print(f"Jumlah baris terdeteksi outlier: {outliers_z.shape[0]}")
print(outliers_z[numerical_cols].head())

# Handling Outlier 
def cap_outliers_iqr(data, cols):
    for col in cols:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        data[col] = np.clip(data[col], lower, upper)
    return data

df = cap_outliers_iqr(df, numerical_cols)

z_scores_after = np.abs(stats.zscore(df[numerical_cols]))
outliers_after = df[(z_scores_after > 3).any(axis=1)]
print(f"\nJumlah outlier setelah handling (capping IQR): {outliers_after.shape[0]}")

#3. Transformation
print("\n" + "=" * 60)
print("3. TRANSFORMATION")
print("=" * 60)

#encoding kategorikal
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

target_encoder = LabelEncoder()
df["classification"] = target_encoder.fit_transform(df["classification"])
print("\nMapping label target:",
      dict(zip(target_encoder.classes_, target_encoder.transform(target_encoder.classes_))))

#scaling dengan standardscaler
X = df.drop(columns=["classification"])
y = df["classification"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n--- Hasil Scaling (StandardScaler) ---")
print(pd.DataFrame(X_scaled, columns=X.columns).head())
print(f"\nJumlah fitur setelah transformasi: {X_scaled.shape[1]}")

# 4. DATA SPLITTING 
print("\n" + "=" * 60)
print("4. DATA SPLITTING")
print("=" * 60)

X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y, test_size=0.2, random_state=SEED, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp
)

print(f"Data Training   : {X_train.shape[0]} sampel")
print(f"Data Validation : {X_val.shape[0]} sampel")
print(f"Data Testing    : {X_test.shape[0]} sampel")

n_features = X_train.shape[1]
X_train_seq = X_train.reshape(X_train.shape[0], n_features, 1)
X_val_seq = X_val.reshape(X_val.shape[0], n_features, 1)
X_test_seq = X_test.reshape(X_test.shape[0], n_features, 1)