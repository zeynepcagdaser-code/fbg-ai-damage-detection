import streamlit as st
import numpy as np
import os
import sys
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from collections import Counter

plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams.update({
    'figure.dpi': 120,
    'font.size': 12,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'lines.linewidth': 2,
})

WINDOW_SIZE = 32
STRIDE = 8
MODEL_PATH = 'models/fbg_team_1dcnn.keras'
RAW_LABELS = ['normal', 'mild_damage', 'severe_damage']
DISPLAY_LABELS = {
    'normal': 'Normal',
    'mild_damage': 'Hafif Hasar',
    'severe_damage': 'Ağır Hasar'
}

st.set_page_config(
    page_title='FBG Sensörlerde AI ile Hasar Tespiti',
    page_icon='📊',
    layout='wide',
)

st.title('FBG Sensörlerde Yapay Zekâ ile Hasar Tespiti')
st.markdown(
    'Bu panel, Simay ve Aleyna’dan gelen FBG zaman serisi verilerini işleyerek CNN ve LSTM tabanlı yapay zekâ modelleriyle hasar tespiti yapar.'
)


def safe_load_model(model_path=MODEL_PATH):
    try:
        candidates = [model_path]
        if not os.path.isabs(model_path):
            candidates.append(str(Path(__file__).parent / model_path))

        for candidate in candidates:
            if os.path.exists(candidate):
                return tf.keras.models.load_model(candidate, compile=False)

        # Legacy: varsa .h5 modeli otomatik .keras'a çevir
        for candidate in candidates:
            h5_path = os.path.splitext(candidate)[0] + ".h5"
            if os.path.exists(h5_path):
                legacy_model = tf.keras.models.load_model(h5_path)
                os.makedirs(os.path.dirname(candidate) or ".", exist_ok=True)
                legacy_model.save(candidate)
                return tf.keras.models.load_model(candidate, compile=False)
    except Exception:
        return None
    return None


def get_window_params_from_model(model):
    window_size = WINDOW_SIZE
    stride = STRIDE
    try:
        if model is not None and hasattr(model, "input_shape"):
            shape = model.input_shape
            if isinstance(shape, tuple) and len(shape) >= 3 and shape[1]:
                window_size = int(shape[1])
    except Exception:
        pass
    return window_size, stride


def run_team_model_training():
    script_path = Path(__file__).parent / "train_team_1dcnn.py"
    if not script_path.exists():
        return False, f"Eğitim scripti bulunamadı: {script_path}", ""

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(Path(__file__).parent),
            capture_output=True,
            text=True,
            check=False,
        )
        output = (result.stdout or "") + ("\n" + (result.stderr or "") if result.stderr else "")
        if result.returncode != 0:
            return False, "Eğitim sırasında hata oluştu.", output
        return True, "Eğitim tamamlandı ve model kaydedildi.", output
    except Exception as exc:
        return False, f"Eğitim başlatılamadı: {exc}", ""


def moving_average(signal, window=5):
    signal = np.asarray(signal, dtype=float)
    if len(signal) < window:
        return signal
    kernel = np.ones(window) / window
    return np.convolve(signal, kernel, mode='same')


def normalize_signal(signal):
    signal = np.asarray(signal, dtype=float)
    min_val = np.nanmin(signal)
    max_val = np.nanmax(signal)
    if np.isclose(max_val, min_val):
        return np.zeros_like(signal)
    return (signal - min_val) / (max_val - min_val)


def close_fig(fig):
    try:
        plt.close(fig)
    except Exception:
        pass


def read_csv_flexible(path: Path) -> pd.DataFrame:
    last_exc = None
    for encoding in ("utf-8-sig", "utf-8", "cp1254", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except Exception as exc:
            last_exc = exc
            continue
    raise last_exc  # type: ignore[misc]


def normalize_project_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    lower_map = {str(col).strip().lower(): str(col).strip() for col in df.columns}

    time_col = lower_map.get("time") or lower_map.get("zaman")
    if time_col is None:
        # bazı dosyalarda zaman sütunu olmayabilir; index üzerinden üret
        df["time"] = np.arange(len(df), dtype=float)
    else:
        df = df.rename(columns={time_col: "time"})

    noisy_col = lower_map.get("delta_lambda_noisy")
    if noisy_col is not None and noisy_col != "delta_lambda_noisy":
        df = df.rename(columns={noisy_col: "delta_lambda_noisy"})

    filtered_col = lower_map.get("delta_lambda_filtered")
    if filtered_col is not None and filtered_col != "delta_lambda_filtered":
        df = df.rename(columns={filtered_col: "delta_lambda_filtered"})

    label_col = lower_map.get("label") or lower_map.get("etiket")
    if label_col is not None and label_col != "label":
        df = df.rename(columns={label_col: "label"})

    if "delta_lambda_noisy" in df.columns:
        df["delta_lambda_noisy"] = pd.to_numeric(df["delta_lambda_noisy"], errors="coerce")
    if "delta_lambda_filtered" in df.columns:
        df["delta_lambda_filtered"] = pd.to_numeric(df["delta_lambda_filtered"], errors="coerce")
    if "time" in df.columns:
        df["time"] = pd.to_numeric(df["time"], errors="coerce")
        if df["time"].isna().any():
            df["time"] = df["time"].ffill().bfill()

    if "label" in df.columns:
        df["label"] = df["label"].astype(str).str.lower().str.strip()
        df["label"] = df["label"].map(standardize_label)

    return df


def standardize_label(value: str):
    if value is None:
        return value
    text = str(value).strip().lower()
    text = text.replace("ı", "i").replace("ğ", "g").replace("ü", "u").replace("ş", "s").replace("ö", "o").replace("ç", "c")

    if text in ("normal", "0"):
        return "normal"
    if "mild" in text or "hafif" in text or text in ("1", "mild_damage", "mild damage"):
        return "mild_damage"
    if "severe" in text or "agir" in text or "agır" in text or "ağir" in text or "ağır" in text or text in ("2", "severe_damage", "severe damage"):
        return "severe_damage"
    return text


def find_first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None


def load_simay_and_aleyna_data():
    base_dir = Path(__file__).parent
    simay_candidates = [
        base_dir / "data" / "fbg_simulink_labeled_dataset SİMAY.csv",
        base_dir / "fbg_simulink_labeled_dataset SİMAY.csv",
        base_dir / "fbg_simulink_labeled_dataset (1).csv",
    ]
    aleyna_candidates = [
        base_dir / "data" / "fbg_filtered_dataset ALEYNA.csv",
        base_dir / "fbg_filtered_dataset ALEYNA.csv",
        base_dir / "data" / "fbg_filtered_dataset.csv",
        base_dir / "fbg_filtered_dataset (1).csv",
        base_dir / "data" / "fbg_filtered_dataset (1).csv",
    ]

    simay_path = find_first_existing(simay_candidates)
    aleyna_path = find_first_existing(aleyna_candidates)

    if simay_path is None or aleyna_path is None:
        return None, None, simay_path, aleyna_path

    simay_df = normalize_project_columns(read_csv_flexible(simay_path))
    aleyna_df = normalize_project_columns(read_csv_flexible(aleyna_path))

    if "delta_lambda_filtered" not in aleyna_df.columns or aleyna_df["delta_lambda_filtered"].isna().all():
        if "delta_lambda_noisy" in aleyna_df.columns:
            aleyna_df["delta_lambda_filtered"] = moving_average(
                aleyna_df["delta_lambda_noisy"].astype(float).values, window=5
            )

    simay_df = simay_df.dropna(subset=["time", "delta_lambda_noisy"]).reset_index(drop=True)
    if "delta_lambda_filtered" in aleyna_df.columns:
        aleyna_df = aleyna_df.dropna(subset=["time", "delta_lambda_noisy", "delta_lambda_filtered"]).reset_index(drop=True)
    else:
        aleyna_df = aleyna_df.dropna(subset=["time", "delta_lambda_noisy"]).reset_index(drop=True)

    return simay_df, aleyna_df, simay_path, aleyna_path


def load_uploaded_data(uploaded_file):
    if uploaded_file is None:
        return None, 'Lütfen CSV veya XLSX uzantılı bir dosya yükleyin.'

    try:
        if uploaded_file.name.lower().endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
    except Exception as exc:
        return None, f'Dosya okunurken hata oluştu: {exc}'

    if df.empty:
        return None, 'Yüklenen dosya boş.'

    df.columns = [str(col).strip() for col in df.columns]
    lower_map = {col.lower(): col for col in df.columns}

    if 'time' not in lower_map:
        return None, "'time' sütunu bulunamadı."
    if 'delta_lambda_noisy' not in lower_map:
        return None, "'delta_lambda_noisy' sütunu bulunamadı."

    mapped = {
        'time': lower_map['time'],
        'delta_lambda_noisy': lower_map['delta_lambda_noisy']
    }

    if 'delta_lambda_filtered' in lower_map:
        mapped['delta_lambda_filtered'] = lower_map['delta_lambda_filtered']
    if 'label' in lower_map:
        mapped['label'] = lower_map['label']

    columns = [mapped['time'], mapped['delta_lambda_noisy']]
    if 'delta_lambda_filtered' in mapped:
        columns.append(mapped['delta_lambda_filtered'])
    if 'label' in mapped:
        columns.append(mapped['label'])

    df = df[columns]
    rename_map = {
        mapped['time']: 'time',
        mapped['delta_lambda_noisy']: 'delta_lambda_noisy'
    }
    if 'delta_lambda_filtered' in mapped:
        rename_map[mapped['delta_lambda_filtered']] = 'delta_lambda_filtered'
    if 'label' in mapped:
        rename_map[mapped['label']] = 'label'

    df = df.rename(columns=rename_map)
    df['delta_lambda_noisy'] = pd.to_numeric(df['delta_lambda_noisy'], errors='coerce')
    if 'delta_lambda_filtered' in df.columns:
        df['delta_lambda_filtered'] = pd.to_numeric(df['delta_lambda_filtered'], errors='coerce')
    if 'label' in df.columns:
        df['label'] = df['label'].astype(str).str.lower().str.strip().map(standardize_label)

    df = df.dropna(subset=['delta_lambda_noisy'])
    if df.empty:
        return None, 'Yüklenen veri içinde geçerli delta_lambda_noisy değeri bulunamadı.'

    if 'delta_lambda_filtered' not in df.columns or df['delta_lambda_filtered'].isna().all():
        df['delta_lambda_filtered'] = moving_average(df['delta_lambda_noisy'].values, window=5)
    else:
        df['delta_lambda_filtered'] = df['delta_lambda_filtered'].ffill().bfill()
        if df['delta_lambda_filtered'].isna().any():
            df['delta_lambda_filtered'] = moving_average(df['delta_lambda_noisy'].values, window=5)

    if 'label' in df.columns:
        valid_labels = set(RAW_LABELS)
        invalid = df.loc[~df['label'].isin(valid_labels), 'label'].unique()
        if len(invalid) > 0:
            return None, f'Geçersiz label değerleri bulundu: {list(invalid)}. Kullanılabilir: {RAW_LABELS}.'

    return df.reset_index(drop=True), None


def create_windows(df, window_size=WINDOW_SIZE, stride=STRIDE):
    filtered = df['delta_lambda_filtered'].astype(float).values
    noisy = df['delta_lambda_noisy'].astype(float).values

    if len(filtered) < window_size:
        return None, None, None, f"Veri uzunluğu {window_size}'ten az. Lütfen daha uzun bir zaman serisi yükleyin."

    filtered_norm = normalize_signal(filtered)
    noisy_norm = normalize_signal(noisy)

    X = []
    X_noisy = []
    window_labels = []
    has_label = 'label' in df.columns

    for start in range(0, len(filtered_norm) - window_size + 1, stride):
        X.append(filtered_norm[start:start + window_size])
        X_noisy.append(noisy_norm[start:start + window_size])

        if has_label:
            window_segment = df['label'].iloc[start:start + window_size].astype(str)
            most_common = Counter(window_segment).most_common(1)[0][0]
            window_labels.append(most_common)

    X = np.array(X, dtype=np.float32)[..., np.newaxis]
    X_noisy = np.array(X_noisy, dtype=np.float32)

    if has_label:
        return X, X_noisy, window_labels, None
    return X, X_noisy, None, None


def predict_windows(model, X):
    probabilities = model.predict(X, batch_size=32, verbose=0)
    indices = np.argmax(probabilities, axis=1)
    labels = [DISPLAY_LABELS[RAW_LABELS[idx]] for idx in indices]
    return indices, labels, probabilities


def generate_summary(predicted_labels):
    counts = Counter(predicted_labels)
    if not counts:
        return None, 0.0
    label, count = counts.most_common(1)[0]
    score = count / len(predicted_labels) * 100
    return label, score


def plot_signal_analysis(df):
    filtered_norm = normalize_signal(df['delta_lambda_filtered'].astype(float).values)
    noisy_norm = normalize_signal(df['delta_lambda_noisy'].astype(float).values)
    x_axis = np.arange(len(filtered_norm))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        x_axis,
        noisy_norm,
        label='Gürültülü Sinyal (delta_lambda_noisy)',
        color='red',
        alpha=0.5,
        linestyle='--',
        linewidth=2,
    )
    ax.plot(
        x_axis,
        filtered_norm,
        label='Filtrelenmiş Sinyal (delta_lambda_filtered)',
        color='darkgreen',
        linewidth=3,
    )
    ax.set_xlabel('Zaman İndeksi', fontsize=12)
    ax.set_ylabel('Normalize Delta Lambda', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=11)
    ax.set_title('FBG Zaman Serisi Analizi', fontsize=14, fontweight='bold')
    return fig


def plot_probability_chart(probabilities, predicted_index):
    class_names = [DISPLAY_LABELS[label] for label in RAW_LABELS]
    values = probabilities * 100
    colors = ['#ef4444' if idx == predicted_index else '#2563eb' for idx in range(len(values))]

    fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
    bars = ax.bar(class_names, values, color=colors, alpha=0.92, edgecolor='none', width=0.55)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1.5,
            f'{height:.1f}%',
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='bold',
            color='#0f172a'
        )

    ax.set_ylim(0, 100)
    ax.set_ylabel('Olasılık (%)', fontsize=12)
    ax.set_title('Sınıf Olasılıkları (%)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return fig


def build_confusion_matrix(y_true, y_pred):
    y_true_idx = [RAW_LABELS.index(label) for label in y_true]
    y_pred_idx = [RAW_LABELS.index(label) for label in y_pred]
    return confusion_matrix(y_true_idx, y_pred_idx, labels=[0, 1, 2])


def format_classification_report(y_true, y_pred):
    y_true_idx = [RAW_LABELS.index(label) for label in y_true]
    y_pred_idx = [RAW_LABELS.index(label) for label in y_pred]
    report = classification_report(
        y_true_idx,
        y_pred_idx,
        target_names=[DISPLAY_LABELS[label] for label in RAW_LABELS],
        output_dict=True,
        zero_division=0,
    )
    return pd.DataFrame(report).transpose()


def render_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(5.4, 4), dpi=120)
    im = ax.imshow(cm, cmap='Blues')
    labels = [DISPLAY_LABELS[label] for label in RAW_LABELS]
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=12)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel('Tahmin Edilen', fontsize=12)
    ax.set_ylabel('Gerçek', fontsize=12)
    ax.set_title('Karmaşıklık Matrisi (Confusion Matrix)', fontsize=14, fontweight='bold')

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = 'white' if cm[i, j] > thresh else 'black'
            ax.text(j, i, cm[i, j], ha='center', va='center', color=color, fontsize=13, fontweight='bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def render_dashboard_intro():
    st.markdown('''
    <style>
    .hero-card {background: linear-gradient(135deg, #eff6ff 0%, #ecfdf5 100%); border-radius: 24px; padding: 28px; margin-bottom: 24px; border: 1px solid rgba(15, 23, 42, 0.08);}
    .hero-card h2 {margin: 0 0 10px 0;}
    .hero-card p {margin: 8px 0 0; color: #334155; font-size: 1rem; line-height:1.65;}
    .mini-badge {display:inline-block; background:#2563eb; color:white; padding:8px 14px; border-radius:999px; font-size:0.95rem; margin-right:8px;}
    </style>
    ''', unsafe_allow_html=True)

    st.markdown(
        """
        <div class='hero-card'>
            <div style='display:flex; flex-wrap:wrap; align-items:center; gap:14px;'>
                <div style='flex:1; min-width:260px;'>
                    <h2 style='font-size:2rem; margin-bottom:10px;'>FBG Sensörlerde Yapay Zekâ ile Hasar Tespiti</h2>
                    <p>Bu panel, Simay ve Aleyna’dan gelen FBG zaman serisi verilerini işleyerek CNN ve LSTM tabanlı yapay zekâ modelleriyle hasar tespiti yapar.</p>
                </div>
                <div style='display:flex; gap:10px; flex-wrap:wrap;'>
                    <span class='mini-badge'>3 Sınıf AI Modeli</span>
                    <span class='mini-badge'>1D CNN + LSTM</span>
                    <span class='mini-badge'>Gerçek Veri Grafikleri</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pipeline_timeline():
    steps = [
        {
            'title': '1️⃣ Fiziksel Modelleme',
            'owner': 'Simay',
            'description': 'Simulink tabanlı FBG zaman serisi üretimi.',
            'color': '#e9f5ff',
        },
        {
            'title': '2️⃣ Gürültü Temizleme',
            'owner': 'Aleyna',
            'description': 'delta_lambda_filtered üretimi (moving average).',
            'color': '#ecfdf5',
        },
        {
            'title': '3️⃣ Feature Engineering',
            'owner': 'Gizem',
            'description': 'Min-Max scaling + windowing (32) + stride (8).',
            'color': '#f5f3ff',
        },
        {
            'title': '4️⃣ CNN Modeli',
            'owner': 'Zeynep',
            'description': '1D CNN ile hasar sınıflandırması (3 sınıf).',
            'color': '#eff6ff',
        },
        {
            'title': '5️⃣ LSTM Modeli',
            'owner': 'Emine',
            'description': 'LSTM tabanlı zaman serisi sınıflandırması (planlı).',
            'color': '#f3e8ff',
        },
        {
            'title': '6️⃣ Hasar Analizi',
            'owner': 'Çağla',
            'description': 'Hasar oranı + güven yüzdesi + kısa yorum.',
            'color': '#fff7ed',
        },
        {
            'title': '7️⃣ Entegrasyon & Dashboard',
            'owner': 'Zeynep & Emine',
            'description': 'Kullanıcıdan veri alıp inference yapan panel.',
            'color': '#f0f9ff',
        },
    ]

    html = '<div style="display:flex; flex-wrap:wrap; gap:14px; align-items:flex-start; justify-content:space-between;">'
    for index, step in enumerate(steps):
        html += f"<div style='flex:1 1 180px; min-width:180px; border-radius:18px; padding:20px; background:{step['color']}; border:1px solid rgba(15, 23, 42, 0.08); box-shadow:0 2px 10px rgba(15, 23, 42, 0.05);'>"
        html += f"<div style='font-size:18px; font-weight:700; margin-bottom:6px;'>{step['title']}</div>"
        html += f"<div style='font-size:14px; color:#0f172a; margin-bottom:10px; font-weight:600;'>Sorumlu: {step['owner']}</div>"
        html += f"<div style='font-size:13px; line-height:1.6; color:#334155;'>{step['description']}</div>"
        html += '</div>'
        if index < len(steps) - 1:
            html += "<div style='display:flex; align-items:center; justify-content:center; min-width:40px; font-size:20px; color:#0f172a;'>→</div>"
    html += '</div>'

    st.markdown(html, unsafe_allow_html=True)


def summarize_metrics(y_true, y_pred):
    y_true_idx = [RAW_LABELS.index(label) for label in y_true]
    y_pred_idx = [RAW_LABELS.index(label) for label in y_pred]
    accuracy = accuracy_score(y_true_idx, y_pred_idx)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_idx,
        y_pred_idx,
        labels=[0, 1, 2],
        zero_division=0,
    )
    return accuracy, precision, recall, f1


def plot_simay_raw(simay_df: pd.DataFrame):
    x = simay_df["time"].astype(float).values
    y = simay_df["delta_lambda_noisy"].astype(float).values
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(x, y, color="#b91c1c", linewidth=2.0, alpha=0.95)
    ax.set_title("Simay Tarafından Üretilen Ham FBG Sinyali", fontsize=15, fontweight="bold")
    ax.set_xlabel("Zaman")
    ax.set_ylabel("Delta Lambda")
    ax.grid(alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig


def plot_noisy_vs_filtered(simay_df: pd.DataFrame, aleyna_df: pd.DataFrame):
    x1 = simay_df["time"].astype(float).values
    y_noisy = simay_df["delta_lambda_noisy"].astype(float).values

    x2 = aleyna_df["time"].astype(float).values
    y_filtered = aleyna_df["delta_lambda_filtered"].astype(float).values

    # Farklı uzunluklarda olabilir; görselde taşmaması için min uzunluğa indir
    n = min(len(x1), len(y_noisy), len(x2), len(y_filtered))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(x1[:n], y_noisy[:n], color="#dc2626", linestyle="--", alpha=0.5, linewidth=1.8, label="delta_lambda_noisy")
    ax.plot(x2[:n], y_filtered[:n], color="#166534", linewidth=3.0, label="delta_lambda_filtered")
    ax.set_title("Ham ve Filtrelenmiş FBG Sinyali Karşılaştırması", fontsize=15, fontweight="bold")
    ax.set_xlabel("Zaman")
    ax.set_ylabel("Delta Lambda")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig


def plot_feature_engineering(filtered: np.ndarray):
    filtered = np.asarray(filtered, dtype=float)
    filtered_norm = normalize_signal(filtered)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(filtered_norm, color="#166534", linewidth=2.2, alpha=0.9, label="Filtered (normalize)")
    ax.plot(filtered_norm, color="#2563eb", linewidth=1.6, alpha=0.7, label="Min-Max Scaling")
    ax.set_title("Normalize Edilmiş Veri (Min-Max Scaling)", fontsize=15, fontweight="bold")
    ax.set_xlabel("Örnek İndeksi")
    ax.set_ylabel("Ölçeklenmiş Değer (0-1)")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig


def plot_window_preview(filtered: np.ndarray, start_idx: int = 0):
    filtered = np.asarray(filtered, dtype=float)
    start_idx = int(max(0, min(start_idx, max(0, len(filtered) - WINDOW_SIZE))))
    window = normalize_signal(filtered[start_idx:start_idx + WINDOW_SIZE])
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.plot(window, color="#2563eb", linewidth=2.4)
    ax.fill_between(np.arange(len(window)), window, color="#bfdbfe", alpha=0.4)
    ax.set_ylim(0, 1)
    ax.set_title("Seçilen 32 Noktalık Pencere (Normalize)", fontsize=15, fontweight="bold")
    ax.set_xlabel("Pencere İndeksi")
    ax.set_ylabel("Ölçeklenmiş Değer")
    ax.grid(alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig


def section_header(title, description, source):
    st.markdown(f"""
    <div style='border-left:4px solid #2563eb; padding:16px 18px 12px; margin-bottom:10px;'>
        <div style='font-size:1.15rem; font-weight:700; color:#0f172a; margin-bottom:4px;'>{title}</div>
        <div style='color:#475569; line-height:1.65; margin-bottom:4px;'>{description}</div>
        <div style='color:#334155; font-size:0.92rem; font-weight:600;'>Veri Kaynağı: {source}</div>
    </div>
    """, unsafe_allow_html=True)


def render_info_card(title, details, accent_color='#ffffff'):
    content = ''.join([f"<li style='margin-bottom:4px;'>{item}</li>" for item in details])
    html = f"""
    <div style='border:1px solid rgba(15,23,42,0.12); background:{accent_color}; border-radius:18px; padding:18px; margin-top:8px;'>
        <div style='font-weight:700; font-size:15px; margin-bottom:10px;'>{title}</div>
        <ul style='margin:0; padding-left:18px; color:#334155; font-size:14px;'>{content}</ul>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def st_show_fig(fig):
    st.pyplot(fig, use_container_width=True)
    close_fig(fig)


def plot_training_curves(history_df: pd.DataFrame, title_prefix: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))

    if "loss" in history_df.columns:
        axes[0].plot(history_df["loss"].values, label="loss", color="#2563eb")
    if "val_loss" in history_df.columns:
        axes[0].plot(history_df["val_loss"].values, label="val_loss", color="#dc2626")
    axes[0].set_title(f"{title_prefix} Loss", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.2)
    axes[0].legend()
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    if "accuracy" in history_df.columns:
        axes[1].plot(history_df["accuracy"].values, label="accuracy", color="#2563eb")
    if "val_accuracy" in history_df.columns:
        axes[1].plot(history_df["val_accuracy"].values, label="val_accuracy", color="#16a34a")
    axes[1].set_title(f"{title_prefix} Accuracy", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(alpha=0.2)
    axes[1].legend()
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    fig.tight_layout()
    return fig


def render_simay_section(simay_df: pd.DataFrame, simay_path: Path):
    st.markdown('---')
    with st.container():
        section_header(
            '1. Simay - Fiziksel Modelleme ve Ham FBG Sinyali',
            'Simay, FBG sensör yapısı ve Bragg dalga boyu kaymasını Simulink ortamında modelleyerek ham zaman serisi verisini üretmiştir.',
            simay_path.name,
        )
        col1, col2 = st.columns([3, 1])
        with col1:
            st_show_fig(plot_simay_raw(simay_df))
        with col2:
            render_info_card('Teknik Özet', [
                'X ekseni: zaman / time',
                'Y ekseni: delta_lambda_noisy',
                'Ham sinyal (kırmızı)',
            ], accent_color='#f8fafc')


def render_aleyna_section(simay_df: pd.DataFrame, aleyna_df: pd.DataFrame, simay_path: Path, aleyna_path: Path):
    st.markdown('---')
    with st.container():
        section_header(
            '2. Aleyna - Gürültü Temizleme ve Sinyal Filtreleme',
            'Aleyna, Simay’dan gelen gürültülü FBG sinyaline moving average filtresi uygulayarak AI modeline daha uygun, daha kararlı bir filtrelenmiş sinyal elde etmiştir.',
            f'{simay_path.name} + {aleyna_path.name}',
        )
        col1, col2 = st.columns([3, 1])
        with col1:
            st_show_fig(plot_noisy_vs_filtered(simay_df, aleyna_df))
        with col2:
            render_info_card('Teknik Özet', [
                'delta_lambda_noisy: kırmızı (dashed, alpha=0.5)',
                'delta_lambda_filtered: koyu yeşil (linewidth=3)',
                'Aynı zaman ekseni üzerinde karşılaştırma',
            ], accent_color='#ecfdf5')


def render_gizem_section(aleyna_df: pd.DataFrame, aleyna_path: Path):
    st.markdown('---')
    with st.container():
        section_header(
            '3. Gizem - Ölçekleme, Pencereleme ve Etiketleme',
            'Gizem, filtrelenmiş sinyali makine öğrenmesine uygun hale getirmek için Min-Max scaling, pencereleme (32) ve stride (8) adımlarını uygular.',
            aleyna_path.name,
        )

        filtered = aleyna_df["delta_lambda_filtered"].astype(float).values
        col1, col2 = st.columns([3, 1])
        with col1:
            st_show_fig(plot_feature_engineering(filtered))
        with col2:
            render_info_card('Parametreler', [
                f'window_size = {WINDOW_SIZE}',
                f'stride = {STRIDE}',
                'Min-Max Scaling (0-1)',
                'Label encoding (3 sınıf)',
            ], accent_color='#eef2ff')

        st.caption('Aşağıda örnek bir 32 noktalık pencere gösterilir (filtrelenmiş sinyalden).')
        max_start = max(0, len(filtered) - WINDOW_SIZE)
        start_idx = st.slider('Pencere başlangıç indeksi', 0, max_start, 0, key='gizem_window_start')
        st_show_fig(plot_window_preview(filtered, start_idx=start_idx))


def render_model_metrics_section(model, windows, window_labels, model_name: str):
    predicted_indices, predicted_labels, probabilities = predict_windows(model, windows)

    st.subheader('Model Metrikleri')
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric('Toplam Pencere', f'{len(predicted_labels)}')
    with col2:
        st.metric('Ortalama Güven', f'%{probabilities.max(axis=1).mean() * 100:.1f}')
    with col3:
        detected_label = Counter(predicted_labels).most_common(1)[0][0] if predicted_labels else "-"
        st.metric('Tespit Edilen Durum', detected_label)
    with col4:
        # Hasar oranı: (Hafif + Ağır) / toplam
        total = len(predicted_labels)
        damage_count = sum(1 for lbl in predicted_labels if lbl in ("Hafif Hasar", "Ağır Hasar"))
        damage_ratio = (damage_count / total) * 100 if total else 0.0
        st.metric('Hasar Oranı', f'%{damage_ratio:.1f}')

    if window_labels is None:
        st.info('Bu veri etiketsiz olduğu için doğruluk hesaplanamadı.')
        return predicted_labels, probabilities

    # 80/20 holdout değerlendirme (panelde gerçek test metrikleri)
    y_idx = [RAW_LABELS.index(label) for label in window_labels]
    all_idx = np.arange(len(windows))
    _, test_idx = train_test_split(
        all_idx,
        test_size=0.2,
        random_state=42,
        stratify=y_idx,
    )

    y_true_test = [window_labels[i] for i in test_idx]
    y_pred_test = [RAW_LABELS[predicted_indices[i]] for i in test_idx]

    cm = build_confusion_matrix(y_true_test, y_pred_test)
    metrics_df = format_classification_report(y_true_test, y_pred_test)
    acc, precision, recall, f1 = summarize_metrics(y_true_test, y_pred_test)

    with st.expander("Model Değerlendirme (Test)", expanded=False):
        st.metric('Accuracy (Test)', f'%{acc * 100:.2f}')
        st.metric('Macro F1 (Test)', f'{np.mean(f1):.3f}')

        st.write('**Precision / Recall / F1 (Test)**')
        metric_table = pd.DataFrame({
            'Precision': precision,
            'Recall': recall,
            'F1': f1
        }, index=[DISPLAY_LABELS[label] for label in RAW_LABELS])
        st.dataframe(metric_table.style.format('{:.2f}'), use_container_width=True)

        st.write('**Confusion Matrix (Test)**')
        st_show_fig(render_confusion_matrix(cm))

        st.write('**Classification Report (Test)**')
        st.dataframe(metrics_df.style.format('{:.2f}'), use_container_width=True)

    return predicted_labels, probabilities


def render_cnn_section(aleyna_df: pd.DataFrame):
    st.markdown('---')
    with st.container():
        section_header(
            '4. Zeynep - 1D CNN ile Hasar Tespiti',
            'Zeynep, Gizem’in hazırladığı pencere verilerini kullanarak 1D CNN modeli ile 3 sınıflı hasar tespiti yapar.',
            'models/fbg_team_1dcnn.keras',
        )

        model = safe_load_model("models/fbg_team_1dcnn.keras")
        if model is None:
            st.warning('CNN modeli henüz eğitilmedi. `models/fbg_team_1dcnn.keras` bulunamadı.')
            if st.button('CNN Modelini Eğit', type='primary', key='train_cnn_button'):
                with st.spinner('CNN modeli eğitiliyor... Bu işlem birkaç dakika sürebilir.'):
                    ok, msg, logs = run_team_model_training()
                if ok:
                    st.success(msg)
                    if logs.strip():
                        with st.expander('Eğitim Logları', expanded=False):
                            st.code(logs, language='text')
                    st.rerun()
                else:
                    st.error(msg)
                    if logs.strip():
                        with st.expander('Eğitim Logları', expanded=True):
                            st.code(logs, language='text')
            return None, None, None

        win, stride = get_window_params_from_model(model)
        windows, _, window_labels, err = create_windows(aleyna_df, window_size=win, stride=stride)
        if windows is None:
            st.error(err)
            return model, None, None

        # Eğitim geçmişi (varsa)
        history_path = Path(__file__).parent / "results" / "team_cnn_history.csv"
        dist_path = Path(__file__).parent / "results" / "team_label_distribution.png"
        if dist_path.exists():
            st.subheader('Sınıf Dağılımı (Eğitim Verisi)')
            st.image(str(dist_path), caption='Pencere etiketlerinin sınıf dağılımı', use_container_width=True)

        if history_path.exists():
            try:
                history_df = pd.read_csv(history_path)
                st.subheader('Eğitim Geçmişi')
                st_show_fig(plot_training_curves(history_df, title_prefix="CNN"))
            except Exception:
                st.info('CNN eğitim geçmişi okunamadı.')
        else:
            st.info('CNN eğitim geçmişi bulunamadı. Eğitim sonrası `results/team_cnn_history.csv` üretilir.')

        predicted_labels, probabilities = render_model_metrics_section(model, windows, window_labels, model_name="CNN")
        return model, predicted_labels, probabilities


def render_lstm_section(aleyna_df: pd.DataFrame):
    st.markdown('---')
    with st.container():
        section_header(
            '5. Emine - LSTM ile Zaman Serisi Sınıflandırması',
            'Emine, CNN ile yapılan hasar sınıflandırmasını LSTM mimarisi ile aynı pencere verisi üzerinde tekrar edecektir.',
            'models/fbg_team_lstm.keras',
        )

        lstm_model = safe_load_model("models/fbg_team_lstm.keras")
        if lstm_model is None:
            st.info('LSTM modeli henüz eğitilmedi. Emine tarafından geliştirilecek.')
            return None, None

        win, stride = get_window_params_from_model(lstm_model)
        windows, _, window_labels, err = create_windows(aleyna_df, window_size=win, stride=stride)
        if windows is None:
            st.error(err)
            return lstm_model, None

        history_path = Path(__file__).parent / "results" / "team_lstm_history.csv"
        if history_path.exists():
            try:
                history_df = pd.read_csv(history_path)
                st.subheader('Eğitim Geçmişi')
                st_show_fig(plot_training_curves(history_df, title_prefix="LSTM"))
            except Exception:
                st.info('LSTM eğitim geçmişi okunamadı.')
        else:
            st.info('LSTM eğitim geçmişi bulunamadı (varsa: `results/team_lstm_history.csv`).')

        predicted_labels, probabilities = render_model_metrics_section(lstm_model, windows, window_labels, model_name="LSTM")
        return predicted_labels, probabilities


def render_damage_analysis_section(predicted_labels, probabilities):
    st.markdown('---')
    with st.container():
        section_header(
            '6. Çağla - Hasar Oranı ve Sonuç Analizi',
            'Çağla, model çıktılarından hasar seviyesini yorumlayarak sonuçları mühendislik açısından değerlendirir.',
            'CNN/LSTM tahminleri',
        )

        if predicted_labels is None or probabilities is None:
            st.info('Hasar analizi için önce CNN (veya LSTM) model çıktısı gereklidir.')
            return

        # Etiketleri DISPLAY_LABELS formatına çevir
        mapped = [DISPLAY_LABELS.get(lbl, lbl) for lbl in predicted_labels]
        mapped_counts = pd.Series(mapped).value_counts()
        total = int(mapped_counts.sum())

        dominant = mapped_counts.idxmax()
        confidence = float(probabilities.max(axis=1).mean())

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric('Genel Sonuç', dominant)
        with col2:
            st.metric('Ortalama Güven', f'%{confidence * 100:.1f}')
        with col3:
            severe_ratio = (mapped_counts.get('Ağır Hasar', 0) / total) * 100 if total else 0
            st.metric('Ağır Hasar Oranı', f'%{severe_ratio:.1f}')

        dist_df = pd.DataFrame({
            'Sınıf': ['Normal', 'Hafif Hasar', 'Ağır Hasar'],
            'Oran (%)': [
                (mapped_counts.get('Normal', 0) / total) * 100 if total else 0,
                (mapped_counts.get('Hafif Hasar', 0) / total) * 100 if total else 0,
                (mapped_counts.get('Ağır Hasar', 0) / total) * 100 if total else 0,
            ],
            'Pencere Sayısı': [
                int(mapped_counts.get('Normal', 0)),
                int(mapped_counts.get('Hafif Hasar', 0)),
                int(mapped_counts.get('Ağır Hasar', 0)),
            ],
        })
        st.subheader('Sınıf Dağılımı')
        st.dataframe(dist_df.style.format({'Oran (%)': '{:.1f}'}), use_container_width=True)

        comment_map = {
            'Normal': 'Sistem normal davranış göstermektedir.',
            'Hafif Hasar': 'Erken seviye bozulma belirtisi gözlenmiştir.',
            'Ağır Hasar': 'Kritik hasar paterni tespit edilmiştir.',
        }
        st.subheader('Kısa Yorum')
        st.success(comment_map.get(dominant, 'Sonuç yorumlanamadı.'))


def render_upload_section(model):
    st.markdown('---')
    st.subheader('Kendi FBG Verinizi Yükleyin')
    st.write('CSV veya XLSX dosyası yükleyin. Beklenen sütunlar:')
    st.markdown(
        '''
- time veya zaman  
- delta_lambda_noisy  
- opsiyonel delta_lambda_filtered  
- opsiyonel label veya etiket
        '''
    )

    sample_csv = 'time,delta_lambda_noisy,delta_lambda_filtered,label\n0,6.27,6.30,normal\n1,6.20,6.29,normal\n2,6.16,6.25,mild_damage\n3,6.10,6.19,severe_damage\n'
    st.download_button(
        'Örnek Veri Formatı İndir',
        sample_csv,
        file_name='fbg_sample_format.csv',
        mime='text/csv',
        key='download_sample_format_upload'
    )

    uploaded_file = st.file_uploader(
        'FBG zaman serisi dosyası yükleyin',
        type=['csv', 'xlsx', 'xls'],
        key='main_file_uploader'
    )

    if uploaded_file is None:
        st.info('Veri yüklemesi yaparak modeli doğrudan kendi FBG veriniz üzerinde test edebilirsiniz.')
        return

    st.write(f'**Dosya Adı:** {uploaded_file.name}')
    df, error = load_uploaded_data(uploaded_file)
    if error:
        st.error(error)
        return

    st.write(f'**Satır Sayısı:** {len(df)}')
    st.write('**Sütunlar:**', list(df.columns))

    if model is None:
        st.warning('Henüz eğitilmiş model bulunamadı.\nLütfen önce modeli eğitin.')
        if st.button('CNN Modelini Eğit', type='primary', key='train_team_model_button'):
            with st.spinner('Model eğitiliyor... Bu işlem birkaç dakika sürebilir.'):
                ok, msg, logs = run_team_model_training()

            if ok:
                st.success(msg)
                if logs.strip():
                    with st.expander('Eğitim Logları', expanded=False):
                        st.code(logs, language='text')
                st.rerun()
            else:
                st.error(msg)
                if logs.strip():
                    with st.expander('Eğitim Logları', expanded=True):
                        st.code(logs, language='text')
        return

    win, stride = get_window_params_from_model(model)
    windows, noisy_windows, window_labels, err = create_windows(df, window_size=win, stride=stride)
    if windows is None:
        st.error(err)
        return

    st.success(f'{len(windows)} adet pencere oluşturuldu.')
    st.subheader('Yüklenen Verinin AI Analizi')
    st_show_fig(plot_signal_analysis(df))

    predicted_indices, predicted_labels, probabilities = predict_windows(model, windows)
    summary_label, summary_score = generate_summary(predicted_labels)
    st.markdown(f'### Genel Sonuç: {summary_label} (%{summary_score:.1f})')

    if 'label' in df.columns:
        y_pred = [RAW_LABELS[idx] for idx in predicted_indices]
        cm = build_confusion_matrix(window_labels, y_pred)
        metrics_df = format_classification_report(window_labels, y_pred)
        acc, precision, recall, f1 = summarize_metrics(window_labels, y_pred)

        st.metric('Doğruluk', f'%{acc * 100:.2f}')
        st.write('**Precision / Recall / F1**')
        metric_table = pd.DataFrame({
            'Precision': precision,
            'Recall': recall,
            'F1': f1
        }, index=[DISPLAY_LABELS[label] for label in RAW_LABELS])
        st.dataframe(metric_table.style.format('{:.2f}'))

        st.write('**Confusion Matrix**')
        st_show_fig(render_confusion_matrix(cm))

        st.write('**Classification Report**')
        st.dataframe(metrics_df.style.format('{:.2f}'))
    else:
        st.info('Gerçek etiket olmadığı için doğruluk hesaplanamadı.')

    st.markdown('---')
    st.subheader('Pencere Bazlı Tahminler')
    result_df = pd.DataFrame({
        'Pencere': np.arange(len(predicted_labels)) + 1,
        'Tahmin': predicted_labels,
        'Güven (%)': (probabilities.max(axis=1) * 100).round(1)
    })
    if 'label' in df.columns:
        result_df['Gerçek'] = [DISPLAY_LABELS[label] for label in window_labels]
    st.dataframe(result_df, use_container_width=True)

    st.markdown('---')
    st.subheader('Seçilen Pencerenin Olasılık Grafiği')
    selected_index = st.slider(
        'Pencere seçin',
        0,
        len(predicted_labels) - 1,
        0,
        key='window_index_slider'
    )
    st_show_fig(plot_probability_chart(probabilities[selected_index], predicted_indices[selected_index]))

    st.markdown('---')
    st.subheader('Seçilen Pencere Detayları')
    window_label = window_labels[selected_index] if 'label' in df.columns else None
    pred_label = predicted_labels[selected_index]
    row1, row2 = st.columns(2)
    with row1:
        st.write(f'**Pencere #{selected_index + 1} Tahmini:** {pred_label}')
        st.write(f'**Güven:** %{probabilities[selected_index].max() * 100:.1f}')
    with row2:
        if window_label is not None:
            actual_label = DISPLAY_LABELS.get(window_label, window_label)
            st.write(f'**Gerçek Etiket:** {actual_label}')
            match = actual_label == pred_label
            badge_color = '#16a34a' if match else '#dc2626'
            st.markdown(
                f"<span style='color: white; background-color: {badge_color}; padding: 8px 14px; border-radius: 999px; font-weight:bold;'>{'Doğru Tahmin' if match else 'Yanlış Tahmin'}</span>",
                unsafe_allow_html=True,
            )
        else:
            st.write('Gerçek etiket yok.')


render_dashboard_intro()

st.markdown('### 1. PROJE PIPELINE AKIŞI')
st.write('Ekip görev dağılımı ve projenin ana aşamaları aşağıdaki gibi ilerler.')
render_pipeline_timeline()

simay_df, aleyna_df, simay_path, aleyna_path = load_simay_and_aleyna_data()

if simay_df is None or aleyna_df is None or simay_path is None or aleyna_path is None:
    missing = []
    if simay_path is None:
        missing.append('Simay verisi (fbg_simulink_labeled_dataset ... .csv)')
    if aleyna_path is None:
        missing.append('Aleyna verisi (fbg_filtered_dataset ... .csv)')
    st.warning('Gerçek proje verilerine erişilemiyor. Lütfen veri dosyalarının mevcut olduğunu kontrol edin: ' + ', '.join(missing))
else:
    render_simay_section(simay_df, simay_path)
    render_aleyna_section(simay_df, aleyna_df, simay_path, aleyna_path)
    render_gizem_section(aleyna_df, aleyna_path)

    cnn_model, cnn_pred_labels, cnn_probs = render_cnn_section(aleyna_df)
    lstm_pred_labels, lstm_probs = render_lstm_section(aleyna_df)

    # Öncelik: LSTM varsa onu, yoksa CNN'i analiz et
    if lstm_pred_labels is not None and lstm_probs is not None:
        render_damage_analysis_section(lstm_pred_labels, lstm_probs)
    else:
        render_damage_analysis_section(cnn_pred_labels, cnn_probs)

model = safe_load_model(MODEL_PATH)
render_upload_section(model)
