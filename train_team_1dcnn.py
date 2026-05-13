"""Takım verisi için 1D CNN eğitimi

Bu script, time-series biçimindeki delta_lambda_filtered verisini kullanarak
pencereleme ile örnekler oluşturur ve 1D CNN ile sınıflandırma yapar.
"""

import os
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks


PROJECT_ROOT = Path(__file__).parent
TEAM_DATA_PATHS = [PROJECT_ROOT / "data" / "fbg_filtered_dataset.csv",
                   PROJECT_ROOT / "fbg_filtered_dataset (1).csv"]
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "fbg_team_1dcnn.keras"
RESULTS_DIR = PROJECT_ROOT / "results"


def find_team_data_path():
    for path in TEAM_DATA_PATHS:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Takım veri dosyası bulunamadı. Lütfen 'data/fbg_filtered_dataset.csv' veya 'fbg_filtered_dataset (1).csv' yolunu kontrol edin."
    )


def load_team_timeseries(path):
    df = pd.read_csv(path)
    if 'delta_lambda_filtered' not in df.columns or 'label' not in df.columns:
        raise ValueError("Veri dosyasında 'delta_lambda_filtered' veya 'label' sütunu bulunamadı.")

    series = df['delta_lambda_filtered'].astype(float).values
    labels = df['label'].astype(str).values
    if len(series) < 64:
        raise ValueError("Veri yeterli uzunlukta değil. En az 64 nokta olmalı.")

    return series, labels


def normalize_series(series):
    min_val = np.min(series)
    max_val = np.max(series)
    if max_val - min_val == 0:
        return np.zeros_like(series)
    return (series - min_val) / (max_val - min_val)


def window_series(series, labels, window_size=64, stride=16):
    X = []
    y = []
    for start in range(0, len(series) - window_size + 1, stride):
        window = series[start:start + window_size]
        window_labels = labels[start:start + window_size]
        most_common_label = Counter(window_labels).most_common(1)[0][0]
        X.append(window)
        y.append(most_common_label)
    return np.array(X, dtype=np.float32)[..., np.newaxis], np.array(y, dtype=object)


def build_team_cnn(input_shape=(64, 1), num_classes=3):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(32, kernel_size=3, activation='relu', padding='same'),
        layers.MaxPool1D(pool_size=2),
        layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        layers.MaxPool1D(pool_size=2),
        layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def plot_confusion_matrix(cm, class_names, save_path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title('Confusion Matrix')
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    plt.colorbar(im, ax=ax)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('Gerçek')
    ax.set_xlabel('Tahmin')
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def main():
    print("Takım verisi ile 1D CNN eğitimi başlatılıyor...")

    data_path = find_team_data_path()
    print(f"Veri dosyası: {data_path}")

    series, labels = load_team_timeseries(data_path)
    series = normalize_series(series)

    X, y_raw = window_series(series, labels, window_size=64, stride=16)
    print(f"Pencere sayısı: {len(X)}")

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    class_names = list(encoder.classes_)
    print(f"Sınıf etiketleri: {class_names}")
    print(f"Etiket dağılımı: {np.bincount(y)}")

    if len(X) < 5:
        raise ValueError("Yeterli pencere bulunamadı. Daha uzun bir zaman serisi gerekli.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    print(f"Eğitim örnekleri: {len(X_train)}")
    print(f"Validasyon örnekleri: {len(X_val)}")
    print(f"Test örnekleri: {len(X_test)}")

    model = build_team_cnn(input_shape=(64, 1), num_classes=len(class_names))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    ckpt_path = MODEL_DIR / "fbg_team_1dcnn_best.keras"
    callback_list = [
        callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        callbacks.ModelCheckpoint(str(ckpt_path), monitor='val_loss', save_best_only=True, verbose=1)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=8,
        callbacks=callback_list,
        verbose=2
    )

    print("Model eğitimi tamamlandı.")

    # Eğitim geçmişini kaydet (panel üzerinde gerçek eğitim grafikleri için)
    try:
        hist_df = pd.DataFrame(history.history)
        hist_path = RESULTS_DIR / "team_cnn_history.csv"
        hist_df.to_csv(hist_path, index=False, encoding="utf-8")
        print(f"Eğitim geçmişi kaydedildi: {hist_path}")
    except Exception as exc:
        print(f"Eğitim geçmişi kaydedilemedi: {exc}")

    if ckpt_path.exists():
        model = tf.keras.models.load_model(str(ckpt_path))

    print("Test verisi üzerinde değerlendirme...")
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    report = classification_report(y_test, y_pred, target_names=class_names, digits=4)
    print(report)

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, class_names, RESULTS_DIR / 'team_confusion_matrix.png')
    print(f"Confusion matrix kaydedildi: {RESULTS_DIR / 'team_confusion_matrix.png'}")

    with open(RESULTS_DIR / 'team_classification_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)

    model.save(str(MODEL_PATH))
    print(f"Model başarıyla kaydedildi: {MODEL_PATH}")

    print("Eğitim tamamlandı.")


if __name__ == '__main__':
    main()
