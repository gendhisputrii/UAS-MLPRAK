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

#5. Klasifikasi
#Deep learning 1: CNN
print("\n" + "=" * 60)
print(" MODEL DEEP LEARNING 1: CNN")
print("=" * 60)

def build_cnn(input_shape):
    model = Sequential([
        Conv1D(filters=32, kernel_size=3, activation="relu", input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=64, kernel_size=3, activation="relu", padding="same"),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

cnn_model = build_cnn((n_features, 1))
cnn_model.summary()

early_stop_cnn = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)  # <-- diganti nama variabel

history_cnn = cnn_model.fit(
    X_train_seq, y_train,
    validation_data=(X_val_seq, y_val),
    epochs=50, batch_size=16,
    callbacks=[early_stop_cnn], verbose=1  # <-- diganti
)

#Deep learning 2: LSTM
print("\n" + "=" * 60)
print("MODEL DEEP LEARNING 2: LSTM")
print("=" * 60)

def build_lstm(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

lstm_model = build_lstm((n_features, 1))
lstm_model.summary()

early_stop_lstm = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)  # <-- ditambahkan baru

history_lstm = lstm_model.fit(
    X_train_seq, y_train,
    validation_data=(X_val_seq, y_val),
    epochs=50, batch_size=16,
    callbacks=[early_stop_lstm], verbose=1  # <-- diganti
)

# 6. EVALUASI & KOMPARASI
print("\n" + "=" * 60)
print("7. EVALUASI & KOMPARASI")
print("=" * 60)

def evaluate_model(model, X_test_seq, y_test, model_name):
    y_pred_prob = model.predict(X_test_seq)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n--- Hasil Evaluasi {model_name} ---")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))

    cm = confusion_matrix(y_test, y_pred)
    return {"model": model_name, "accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1, "y_pred": y_pred, "cm": cm}

result_cnn = evaluate_model(cnn_model, X_test_seq, y_test, "CNN")
result_lstm = evaluate_model(lstm_model, X_test_seq, y_test, "LSTM")

comparison_df = pd.DataFrame([
    {"Model": "CNN", "Accuracy": result_cnn["accuracy"], "Precision": result_cnn["precision"],
     "Recall": result_cnn["recall"], "F1-Score": result_cnn["f1"]},
    {"Model": "LSTM", "Accuracy": result_lstm["accuracy"], "Precision": result_lstm["precision"],
     "Recall": result_lstm["recall"], "F1-Score": result_lstm["f1"]},
])
print("\n=== TABEL PERBANDINGAN CNN vs LSTM ===")
print(comparison_df.to_string(index=False))
comparison_df.to_csv("hasil_komparasi_cnn_lstm.csv", index=False)

# Grafik Training Accuracy & Loss 
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(history_cnn.history["accuracy"], label="Train Accuracy")
axes[0, 0].plot(history_cnn.history["val_accuracy"], label="Validation Accuracy")
axes[0, 0].set_title("CNN - Model Accuracy")
axes[0, 0].legend()

axes[0, 1].plot(history_cnn.history["loss"], label="Train Loss")
axes[0, 1].plot(history_cnn.history["val_loss"], label="Validation Loss")
axes[0, 1].set_title("CNN - Model Loss")
axes[0, 1].legend()

axes[1, 0].plot(history_lstm.history["accuracy"], label="Train Accuracy")
axes[1, 0].plot(history_lstm.history["val_accuracy"], label="Validation Accuracy")
axes[1, 0].set_title("LSTM - Model Accuracy")
axes[1, 0].legend()

axes[1, 1].plot(history_lstm.history["loss"], label="Train Loss")
axes[1, 1].plot(history_lstm.history["val_loss"], label="Validation Loss")
axes[1, 1].set_title("LSTM - Model Loss")
axes[1, 1].legend()

plt.tight_layout()
plt.savefig("hasil_training_cnn_lstm.png", dpi=150)
plt.show()

# Confusion Matrix 
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(result_cnn["cm"], annot=True, fmt="d", cmap="Blues",
            xticklabels=target_encoder.classes_, yticklabels=target_encoder.classes_, ax=axes[0])
axes[0].set_title("Confusion Matrix - CNN")

sns.heatmap(result_lstm["cm"], annot=True, fmt="d", cmap="Greens",
            xticklabels=target_encoder.classes_, yticklabels=target_encoder.classes_, ax=axes[1])
axes[1].set_title("Confusion Matrix - LSTM")

plt.tight_layout()
plt.savefig("confusion_matrix_cnn_lstm.png", dpi=150)
plt.show()

# Bar Chart Perbandingan Metrik 
metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
cnn_scores = [result_cnn["accuracy"], result_cnn["precision"], result_cnn["recall"], result_cnn["f1"]]
lstm_scores = [result_lstm["accuracy"], result_lstm["precision"], result_lstm["recall"], result_lstm["f1"]]

x = np.arange(len(metrics))
width = 0.35
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - width/2, cnn_scores, width, label="CNN")
ax.bar(x + width/2, lstm_scores, width, label="LSTM")
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 1.05)
ax.set_title("Perbandingan Performa CNN vs LSTM")
ax.legend()

for i, v in enumerate(cnn_scores):
    ax.text(i - width/2, v + 0.01, f"{v:.2f}", ha="center")
for i, v in enumerate(lstm_scores):
    ax.text(i + width/2, v + 0.01, f"{v:.2f}", ha="center")

plt.tight_layout()
plt.savefig("perbandingan_metrik_cnn_lstm.png", dpi=150)
plt.show()

# Simpan model terbaik 
best_model_name = "CNN" if result_cnn["f1"] >= result_lstm["f1"] else "LSTM"
best_model = cnn_model if best_model_name == "CNN" else lstm_model
best_model.save(f"model_terbaik_{best_model_name}.h5")

print(f"\nModel terbaik berdasarkan F1-Score: {best_model_name}")

# 7. EXPORT UNTUK FLUTTER (TFLite + JSON)
print("\n" + "=" * 60)
print("8. EXPORT UNTUK FLUTTER")
print("=" * 60)

import json

# Convert model terbaik ke TFLite 
converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
tflite_model = converter.convert()

with open("model_ckd.tflite", "wb") as f:
    f.write(tflite_model)

print(f"Model {best_model_name} berhasil dikonversi ke model_ckd.tflite")

# Export parameter StandardScaler ke JSON 
scaler_params = {
    "mean": scaler.mean_.tolist(),
    "scale": scaler.scale_.tolist(),
    "feature_names": list(X.columns)
}
with open("scaler_params.json", "w") as f:
    json.dump(scaler_params, f, indent=2)

print("Scaler params disimpan ke scaler_params.json")

# Export mapping target (label hasil prediksi)
target_mapping = {
    "classes": target_encoder.classes_.tolist()
}
with open("target_mapping.json", "w") as f:
    json.dump(target_mapping, f, indent=2)

# Export mapping kolom kategorikal 
cat_mappings = {}
for col, le in label_encoders.items():
    cat_mappings[col] = dict(zip(le.classes_.tolist(), le.transform(le.classes_).tolist()))

with open("categorical_mappings.json", "w") as f:
    json.dump(cat_mappings, f, indent=2)

print("\n=== Semua file untuk Flutter berhasil diekspor ===")