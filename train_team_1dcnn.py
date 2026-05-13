"""
Takım verisi için 1D CNN eğitimi (LSTM referans pipeline)

Bu script, `delta_lambda_filtered` zaman serisini min-max ölçekler, overlap'li
pencereleme ile örnekler üretir ve 3 sınıflı (normal / mild_damage / severe_damage)
1D CNN modeli eğitir.

Çıktılar:
- models/fbg_team_1dcnn.keras
- results/cnn_training_history.png
- results/cnn_confusion_matrix.png
- results/cnn_classification_report.txt
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    BatchNormalization,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


PROJECT_ROOT = Path(__file__).parent
DATA_CANDIDATES = [
    PROJECT_ROOT / "data" / "fbg_filtered_dataset (1).csv",
    PROJECT_ROOT / "fbg_filtered_dataset (1).csv",
    PROJECT_ROOT / "data" / "fbg_filtered_dataset.csv",
    PROJECT_ROOT / "fbg_filtered_dataset ALEYNA.csv",
]

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "fbg_team_1dcnn.keras"
RESULTS_DIR = PROJECT_ROOT / "results"


@dataclass(frozen=True)
class PipelineConfig:
    window_size: int = 32
    stride: int = 8
    test_size: float = 0.2
    val_size: float = 0.2  # train içinden ayrılır
    random_state: int = 42


def load_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]
    if "delta_lambda_filtered" not in df.columns or "label" not in df.columns:
        raise ValueError("Veri dosyasında 'delta_lambda_filtered' veya 'label' sütunu bulunamadı.")

    df["delta_lambda_filtered"] = pd.to_numeric(df["delta_lambda_filtered"], errors="coerce")
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df.dropna(subset=["delta_lambda_filtered", "label"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("Veri boş veya geçerli 'delta_lambda_filtered' değeri yok.")
    return df


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=float)
    min_val = np.min(signal)
    max_val = np.max(signal)
    if np.isclose(max_val, min_val):
        return np.zeros_like(signal)
    return (signal - min_val) / (max_val - min_val)


def create_windows(
    series: np.ndarray,
    labels: np.ndarray,
    *,
    window_size: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(series) < window_size:
        raise ValueError(f"Veri yeterli uzunlukta değil. En az {window_size} nokta olmalı.")

    X: list[np.ndarray] = []
    y: list[str] = []
    for start in range(0, len(series) - window_size + 1, stride):
        window = series[start : start + window_size]
        window_labels = labels[start : start + window_size]
        most_common_label = Counter(window_labels).most_common(1)[0][0]
        X.append(window)
        y.append(most_common_label)

    X_arr = np.array(X, dtype=np.float32)[..., np.newaxis]
    y_arr = np.array(y, dtype=object)
    return X_arr, y_arr


def prepare_train_test_split(
    X: np.ndarray,
    y_encoded: np.ndarray,
    *,
    test_size: float,
    val_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=test_size,
        stratify=y_encoded,
        random_state=random_state,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_size,
        stratify=y_train,
        random_state=random_state,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_cnn(input_shape: tuple[int, int], num_classes: int) -> Sequential:
    model = Sequential(
        [
            Input(shape=input_shape),
            Conv1D(16, kernel_size=3, activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Conv1D(32, kernel_size=3, activation="relu", padding="same"),
            BatchNormalization(),
            GlobalAveragePooling1D(),
            Dense(32, activation="relu"),
            Dropout(0.35),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def plot_training_history(history_df: pd.DataFrame, save_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    if "loss" in history_df.columns:
        axes[0].plot(history_df["loss"].values, label="loss", color="#2563eb")
    if "val_loss" in history_df.columns:
        axes[0].plot(history_df["val_loss"].values, label="val_loss", color="#dc2626")
    axes[0].set_title("CNN Loss", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.2)
    axes[0].legend()

    if "accuracy" in history_df.columns:
        axes[1].plot(history_df["accuracy"].values, label="accuracy", color="#2563eb")
    if "val_accuracy" in history_df.columns:
        axes[1].plot(history_df["val_accuracy"].values, label="val_accuracy", color="#16a34a")
    axes[1].set_title("CNN Accuracy", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(alpha=0.2)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], save_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
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


def find_dataset_path() -> Path:
    for p in DATA_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Veri dosyası bulunamadı. Beklenen: `data/fbg_filtered_dataset (1).csv` (tercih) veya alternatif adaylar."
    )


def train_and_export(config: PipelineConfig) -> None:
    csv_path = find_dataset_path()
    print(f"Veri dosyası: {csv_path}")

    df = load_dataset(csv_path)
    series = normalize_signal(df["delta_lambda_filtered"].astype(float).values)
    labels = df["label"].astype(str).values

    X, y_raw = create_windows(series, labels, window_size=config.window_size, stride=config.stride)
    print(f"Pencere sayısı: {len(X)} (window={config.window_size}, stride={config.stride})")

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    class_names = list(encoder.classes_)
    print(f"Sınıf etiketleri: {class_names}")
    print(f"Etiket dağılımı: {np.bincount(y)}")

    if len(X) < 15:
        raise ValueError("Pencere sayısı çok az. Daha uzun veri veya daha küçük stride gerekli.")

    X_train, X_val, X_test, y_train, y_val, y_test = prepare_train_test_split(
        X,
        y,
        test_size=config.test_size,
        val_size=config.val_size,
        random_state=config.random_state,
    )

    print(f"Eğitim örnekleri: {len(X_train)} | Validasyon: {len(X_val)} | Test: {len(X_test)}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    ckpt_path = MODEL_DIR / "fbg_team_1dcnn_best.keras"
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        ModelCheckpoint(str(ckpt_path), monitor="val_loss", save_best_only=True, verbose=1),
    ]

    classes = np.unique(y_train)
    class_weights_arr = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = {int(cls): float(w) for cls, w in zip(classes, class_weights_arr)}
    print(f"class_weight: {class_weight}")

    model = build_cnn(input_shape=(config.window_size, 1), num_classes=len(class_names))
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=8,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=2,
    )

    # En iyi ağırlıkları yükle
    if ckpt_path.exists():
        model = load_model(str(ckpt_path))

    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    report_text = classification_report(y_test, y_pred, target_names=class_names, digits=4)
    cm = confusion_matrix(y_test, y_pred)

    # Çıktılar
    model.save(str(MODEL_PATH))
    print(f"Model kaydedildi: {MODEL_PATH}")

    hist_df = pd.DataFrame(history.history)
    plot_training_history(hist_df, RESULTS_DIR / "cnn_training_history.png")
    plot_confusion_matrix(cm, class_names, RESULTS_DIR / "cnn_confusion_matrix.png")

    (RESULTS_DIR / "cnn_classification_report.txt").write_text(report_text, encoding="utf-8")
    print("Çıktılar üretildi:")
    print(f"- {RESULTS_DIR / 'cnn_training_history.png'}")
    print(f"- {RESULTS_DIR / 'cnn_confusion_matrix.png'}")
    print(f"- {RESULTS_DIR / 'cnn_classification_report.txt'}")


def main() -> None:
    config = PipelineConfig()
    train_and_export(config)


if __name__ == "__main__":
    main()
