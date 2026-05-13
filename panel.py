from __future__ import annotations

from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf


# ----------------------------
# Sayfa ayarları / sabitler
# ----------------------------

APP_DIR = Path(__file__).parent
WINDOW_SIZE = 64
STRIDE = 16

MODEL_PATH = APP_DIR / "models" / "fbg_team_1dcnn.keras"

RAW_LABELS = ["normal", "mild_damage", "severe_damage"]
DISPLAY_LABELS = {
    "normal": "Normal",
    "mild_damage": "Hafif Hasar",
    "severe_damage": "Ağır Hasar",
}

plt.style.use("seaborn-v0_8-darkgrid")

st.set_page_config(
    page_title="FBG Sensörlerde Yapay Zekâ ile Hasar Tespiti",
    page_icon="📊",
    layout="wide",
)

st.title("FBG Sensörlerde Yapay Zekâ ile Hasar Tespiti")
st.markdown(
    "Bu panel, Simay ve Aleyna’dan gelen FBG zaman serisi verilerini işleyerek "
    "CNN ve LSTM tabanlı yapay zekâ modelleriyle hasar tespiti yapar."
)


# ----------------------------
# Yardımcılar
# ----------------------------


def close_fig(fig):
    try:
        plt.close(fig)
    except Exception:
        pass


def st_show_fig(fig, *, key: str):
    st.pyplot(fig, use_container_width=True)
    close_fig(fig)


def moving_average(signal: np.ndarray, window: int = 5) -> np.ndarray:
    signal = np.asarray(signal, dtype=float)
    if len(signal) < window:
        return signal
    kernel = np.ones(window) / window
    return np.convolve(signal, kernel, mode="same")


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=float)
    min_val = np.nanmin(signal)
    max_val = np.nanmax(signal)
    if np.isclose(max_val, min_val):
        return np.zeros_like(signal)
    return (signal - min_val) / (max_val - min_val)


def standardize_label(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    text = (
        text.replace("ı", "i")
        .replace("ğ", "g")
        .replace("ü", "u")
        .replace("ş", "s")
        .replace("ö", "o")
        .replace("ç", "c")
    )

    if text in ("normal", "0"):
        return "normal"
    if "mild" in text or "hafif" in text or text in ("1", "mild_damage", "mild damage"):
        return "mild_damage"
    if "severe" in text or "agir" in text or "ağır" in text or text in ("2", "severe_damage", "severe damage"):
        return "severe_damage"
    return text


def read_tabular(uploaded_file) -> pd.DataFrame:
    name = (uploaded_file.name or "").lower()
    if name.endswith((".xls", ".xlsx")):
        return pd.read_excel(uploaded_file)
    return pd.read_csv(uploaded_file)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    lower_map = {str(c).strip().lower(): str(c).strip() for c in df.columns}

    time_col = lower_map.get("time") or lower_map.get("zaman")
    if time_col is None:
        df["time"] = np.arange(len(df), dtype=float)
    else:
        df = df.rename(columns={time_col: "time"})

    noisy_col = lower_map.get("delta_lambda_noisy")
    if noisy_col is None:
        raise ValueError("'delta_lambda_noisy' sütunu bulunamadı.")
    if noisy_col != "delta_lambda_noisy":
        df = df.rename(columns={noisy_col: "delta_lambda_noisy"})

    filtered_col = lower_map.get("delta_lambda_filtered")
    if filtered_col is not None and filtered_col != "delta_lambda_filtered":
        df = df.rename(columns={filtered_col: "delta_lambda_filtered"})

    label_col = lower_map.get("label") or lower_map.get("etiket")
    if label_col is not None and label_col != "label":
        df = df.rename(columns={label_col: "label"})

    df["time"] = pd.to_numeric(df["time"], errors="coerce").ffill().bfill()
    df["delta_lambda_noisy"] = pd.to_numeric(df["delta_lambda_noisy"], errors="coerce")
    if "delta_lambda_filtered" in df.columns:
        df["delta_lambda_filtered"] = pd.to_numeric(df["delta_lambda_filtered"], errors="coerce")

    df = df.dropna(subset=["delta_lambda_noisy"]).reset_index(drop=True)
    if df.empty:
        raise ValueError("Veri içinde geçerli delta_lambda_noisy değeri yok.")

    if "delta_lambda_filtered" not in df.columns or df["delta_lambda_filtered"].isna().all():
        df["delta_lambda_filtered"] = moving_average(df["delta_lambda_noisy"].astype(float).values, window=5)
    else:
        df["delta_lambda_filtered"] = df["delta_lambda_filtered"].ffill().bfill()
        if df["delta_lambda_filtered"].isna().any():
            df["delta_lambda_filtered"] = moving_average(df["delta_lambda_noisy"].astype(float).values, window=5)

    if "label" in df.columns:
        df["label"] = df["label"].astype(str).str.lower().str.strip().map(standardize_label)
        invalid = df.loc[~df["label"].isin(set(RAW_LABELS)), "label"].unique()
        if len(invalid) > 0:
            raise ValueError(f"Geçersiz label değerleri: {list(invalid)}. Kullanılabilir: {RAW_LABELS}.")

    return df


def create_windows(df: pd.DataFrame, window_size: int = WINDOW_SIZE, stride: int = STRIDE):
    filtered = df["delta_lambda_filtered"].astype(float).values

    if len(filtered) < window_size:
        return None, None, "Veri uzunluğu 64'ten az. Lütfen daha uzun zaman serisi yükleyin."

    filtered_norm = normalize_signal(filtered)

    X = []
    y = []
    for start in range(0, len(filtered_norm) - window_size + 1, stride):
        window = filtered_norm[start : start + window_size]
        X.append(window)
        if "label" in df.columns:
            labels = df["label"].iloc[start : start + window_size].astype(str).values
            y.append(Counter(labels).most_common(1)[0][0])

    X = np.array(X, dtype=np.float32)[..., np.newaxis]
    y = np.array(y, dtype=object) if y else None
    return X, y, None


def safe_load_model():
    if not MODEL_PATH.exists():
        return None
    try:
        return tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    except Exception:
        return None


def predict_windows(model, X: np.ndarray):
    probs = model.predict(X, batch_size=32, verbose=0)
    indices = np.argmax(probs, axis=1)
    labels = [DISPLAY_LABELS[RAW_LABELS[i]] for i in indices]
    return indices, labels, probs


def plot_pipeline_cards():
    steps = [
        ("1️⃣ Fiziksel Modelleme", "Simay", "Simulink tabanlı FBG zaman serisi"),
        ("2️⃣ Gürültü Temizleme", "Aleyna", "delta_lambda_filtered üretimi"),
        ("3️⃣ Feature Engineering", "Gizem", "Min-Max scaling + pencereleme"),
        ("4️⃣ CNN Modeli", "Zeynep", "1D CNN hasar sınıflandırması"),
        ("5️⃣ LSTM Modeli", "Emine", "LSTM tabanlı sınıflandırma (planlı)"),
        ("6️⃣ Hasar Analizi", "Çağla", "Hasar oranı + güven + yorum"),
        ("7️⃣ Entegrasyon & Dashboard", "Zeynep & Emine", "Kullanıcıdan veri alıp inference"),
    ]
    html = '<div style="display:flex; flex-wrap:wrap; gap:12px;">'
    for title, owner, out in steps:
        html += (
            "<div style='flex:1 1 220px; min-width:220px; border-radius:16px; padding:16px; "
            "border:1px solid rgba(15,23,42,0.10); background:#ffffff;'>"
            f"<div style='font-weight:800; font-size:16px; margin-bottom:6px;'>{title}</div>"
            f"<div style='color:#334155; font-weight:700; margin-bottom:6px;'>Sorumlu: {owner}</div>"
            f"<div style='color:#475569; font-size:13px; line-height:1.5;'>Çıktı: {out}</div>"
            "</div>"
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def plot_signal(df: pd.DataFrame):
    x = df["time"].astype(float).values
    noisy = df["delta_lambda_noisy"].astype(float).values
    filtered = df["delta_lambda_filtered"].astype(float).values

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, noisy, color="#dc2626", linestyle="--", alpha=0.5, linewidth=1.8, label="delta_lambda_noisy")
    ax.plot(x, filtered, color="#166534", linewidth=3.0, label="delta_lambda_filtered")
    ax.set_title("Ham ve Filtrelenmiş FBG Sinyali", fontweight="bold")
    ax.set_xlabel("Zaman")
    ax.set_ylabel("Delta Lambda")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def render_confusion_matrix(cm: np.ndarray):
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(cm, cmap="Blues")
    labels = [DISPLAY_LABELS[l] for l in RAW_LABELS]
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gerçek")
    ax.set_title("Confusion Matrix", fontweight="bold")

    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


# ----------------------------
# Panel Akışı
# ----------------------------

st.markdown("---")
st.subheader("Proje Pipeline")
plot_pipeline_cards()

st.markdown("---")
st.subheader("Veri Yükleme ve Tahmin")
st.markdown(
    "**Beklenen sütunlar:** `time` veya `zaman`, `delta_lambda_noisy`, opsiyonel `delta_lambda_filtered`, opsiyonel `label` veya `etiket`."
)

sample_csv = (
    "time,delta_lambda_noisy,delta_lambda_filtered,label\n"
    "0,6.27,6.30,normal\n"
    "1,6.20,6.29,normal\n"
    "2,6.16,6.25,mild_damage\n"
    "3,6.10,6.19,severe_damage\n"
)
st.download_button(
    "Örnek Veri Formatı İndir",
    sample_csv,
    file_name="fbg_sample_format.csv",
    mime="text/csv",
    key="download_sample_format",
)

uploaded_file = st.file_uploader(
    "CSV veya XLSX dosyası yükleyin",
    type=["csv", "xlsx", "xls"],
    key="uploader_main",
)

model = safe_load_model()
if model is None:
    st.warning("Model dosyası bulunamadı. Lütfen modeli eğitin veya `models/` klasörüne ekleyin.")
    st.caption(f"Beklenen yol: `{MODEL_PATH.as_posix()}`")

if uploaded_file is None:
    st.info("Veri yükleyince tahmin ve model değerlendirmesi otomatik gösterilir.")
    st.stop()

try:
    raw_df = read_tabular(uploaded_file)
    df = normalize_columns(raw_df)
except Exception as exc:
    st.error(f"Veri okunamadı / doğrulanamadı: {exc}")
    st.stop()

st.write(f"**Satır sayısı:** {len(df)}")
st.write("**Sütunlar:**", list(df.columns))

st_show_fig(plot_signal(df), key="plot_signal")

X, y_window, err = create_windows(df)
if err:
    st.error(err)
    st.stop()

st.success(f"{len(X)} adet pencere oluşturuldu (window={WINDOW_SIZE}, stride={STRIDE}).")

if model is None:
    st.info("Model olmadığı için tahmin yapılamadı.")
    st.stop()

pred_idx, pred_labels, probs = predict_windows(model, X)
dominant, dominant_count = Counter(pred_labels).most_common(1)[0]
dominant_ratio = dominant_count / len(pred_labels) * 100
avg_conf = float(probs.max(axis=1).mean()) if len(probs) else 0.0

st.markdown("### Genel Sonuç")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Tahmin", dominant)
with col2:
    st.metric("Oran", f"%{dominant_ratio:.1f}")
with col3:
    st.metric("Ortalama Güven", f"%{avg_conf * 100:.1f}")

st.markdown("---")
st.subheader("Pencere Bazlı Tahminler")
result_df = pd.DataFrame(
    {
        "Pencere": np.arange(len(pred_labels)) + 1,
        "Tahmin": pred_labels,
        "Güven (%)": (probs.max(axis=1) * 100).round(1),
    }
)
st.dataframe(result_df, use_container_width=True)

if "label" in df.columns:
    st.markdown("---")
    st.subheader("Model Değerlendirmesi (Etiketli Veri)")
    y_true = y_window.tolist() if y_window is not None else []
    y_pred = [RAW_LABELS[i] for i in pred_idx.tolist()]

    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    acc = accuracy_score(y_true, y_pred) if y_true else 0.0
    st.metric("Doğruluk (Accuracy)", f"%{acc * 100:.2f}")

    cm = confusion_matrix(y_true, y_pred, labels=RAW_LABELS)
    st_show_fig(render_confusion_matrix(cm), key="cm_plot")

    report = classification_report(y_true, y_pred, labels=RAW_LABELS, target_names=[DISPLAY_LABELS[x] for x in RAW_LABELS], output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)
else:
    st.info("Etiket olmadığı için doğruluk hesaplanamadı.")
