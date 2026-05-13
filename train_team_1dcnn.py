"""
Bu dosya Zeynep’in 1D CNN model eğitim kodudur.
Emine’nin LSTM modeli için referans olarak kullanılabilir.
"""

from __future__ import annotations

from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    BatchNormalization,
    MaxPooling1D,
    Dropout,
    Flatten,
    Dense,
)


PROJECT_ROOT = Path(__file__).parent
MODEL_PATH = PROJECT_ROOT / "models" / "fbg_team_1dcnn.keras"
RESULTS_DIR = PROJECT_ROOT / "results"

WINDOW_SIZE = 32
STRIDE = 8


def load_X_y() -> tuple[np.ndarray, np.ndarray, list[str]]:
    # Preprocessing "yapılmış" kabul edilse de, referansın tek dosyada çalışması için
    # X/y burada kısa şekilde üretilir: delta_lambda_filtered -> min-max -> windowing.
    candidates = [
        PROJECT_ROOT / "data" / "fbg_filtered_dataset (1).csv",
        PROJECT_ROOT / "fbg_filtered_dataset (1).csv",
        PROJECT_ROOT / "data" / "fbg_filtered_dataset.csv",
        PROJECT_ROOT / "fbg_filtered_dataset ALEYNA.csv",
    ]
    csv_path = next((p for p in candidates if p.exists()), None)
    if csv_path is None:
        raise FileNotFoundError("Veri dosyası bulunamadı: `data/fbg_filtered_dataset (1).csv` bekleniyor.")

    df = pd.read_csv(csv_path)
    if "delta_lambda_filtered" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV içinde `delta_lambda_filtered` ve `label` sütunları olmalı.")

    series = pd.to_numeric(df["delta_lambda_filtered"], errors="coerce").dropna().astype(float).values
    labels = df.loc[df["delta_lambda_filtered"].notna(), "label"].astype(str).str.strip().str.lower().values

    if len(series) < WINDOW_SIZE:
        raise ValueError(f"Veri uzunluğu {WINDOW_SIZE}'ten küçük.")

    min_val, max_val = float(np.min(series)), float(np.max(series))
    series = np.zeros_like(series) if np.isclose(max_val, min_val) else (series - min_val) / (max_val - min_val)

    X = []
    y = []
    for start in range(0, len(series) - WINDOW_SIZE + 1, STRIDE):
        window = series[start : start + WINDOW_SIZE]
        window_labels = labels[start : start + WINDOW_SIZE]
        y.append(Counter(window_labels).most_common(1)[0][0])
        X.append(window)

    X = np.array(X, dtype=np.float32)[..., np.newaxis]  # (N, window_size, 1)
    y = np.array(y, dtype=object)
    class_names = sorted(set(y.tolist()))
    return X, y, class_names


def build_cnn(window_size: int, num_classes: int) -> Sequential:
    model = Sequential(
        [
            Input(shape=(window_size, 1)),
            Conv1D(16, 3, activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling1D(2),
            Dropout(0.25),
            Conv1D(32, 3, activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling1D(2),
            Dropout(0.30),
            Flatten(),
            Dense(64, activation="relu"),
            Dropout(0.35),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def save_training_history(history, save_path: Path) -> None:
    hist = history.history
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(hist.get("loss", []), label="loss", color="#2563eb")
    axes[0].plot(hist.get("val_loss", []), label="val_loss", color="#dc2626")
    axes[0].set_title("CNN Loss", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(alpha=0.2)
    axes[0].legend()

    axes[1].plot(hist.get("accuracy", []), label="accuracy", color="#2563eb")
    axes[1].plot(hist.get("val_accuracy", []), label="val_accuracy", color="#16a34a")
    axes[1].set_title("CNN Accuracy", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(alpha=0.2)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def save_confusion_matrix(cm: np.ndarray, class_names: list[str], save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Karmaşıklık Matrisi (Confusion Matrix)", fontweight="bold")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=12)
    ax.set_yticklabels(class_names, fontsize=12)
    ax.set_xlabel("Tahmin", fontsize=12)
    ax.set_ylabel("Gerçek", fontsize=12)

    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                int(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=13,
                fontweight="bold",
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=240)
    plt.close(fig)


def main() -> None:
    X, y_raw, _ = load_X_y()

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    class_names = list(encoder.classes_)  # normal, mild_damage, severe_damage

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )

    model = build_cnn(window_size=X.shape[1], num_classes=len(class_names))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=8,
        verbose=2,
    )

    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    report_text = classification_report(y_test, y_pred, target_names=class_names, digits=4)
    cm = confusion_matrix(y_test, y_pred)

    model.save(str(MODEL_PATH))

    save_training_history(history, RESULTS_DIR / "cnn_training_history.png")
    save_confusion_matrix(cm, class_names, RESULTS_DIR / "cnn_confusion_matrix.png")
    (RESULTS_DIR / "cnn_classification_report.txt").write_text(report_text, encoding="utf-8")

    print(f"Model kaydedildi: {MODEL_PATH}")
    print(f"Çıktılar: {RESULTS_DIR / 'cnn_training_history.png'}")
    print(f"Çıktılar: {RESULTS_DIR / 'cnn_confusion_matrix.png'}")
    print(f"Çıktılar: {RESULTS_DIR / 'cnn_classification_report.txt'}")


if __name__ == "__main__":
    main()
