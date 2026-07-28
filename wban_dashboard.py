"""
WBAN Anomaly Detection Dashboard
=================================
Streamlit front-end for the Dynamic Two-Stage Hybrid ML Framework described
in the accompanying project report (Section 4). Uploads multi-patient
telemetry CSVs, trains/evaluates the selected Stage-1 gatekeeper(s) (SVM,
Random Forest, Bi-Directional LSTM), applies the shared Stage-2 SVR triage
layer, and renders comparative results.

Run with:
    streamlit run wban_dashboard.py
"""

import io
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
)
from imblearn.over_sampling import RandomOverSampler

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="WBAN Anomaly Detection Dashboard",
    page_icon="\U0001FAC0",
    layout="wide",
)

sns.set_theme(style="whitegrid", palette="muted")

# ==========================================================
# GLOBAL CONFIGURATION (mirrors the project report, Section 3)
# ==========================================================
FEATURES = ["Body_Temperature", "Heart_Rate", "Pulse_Rate", "SpO2", "ECG"]
ROLL_WINDOW = 5
LOOKBACK = 5
TEST_SIZE = 0.20
RANDOM_STATE = 42

CLASS_NAMES = {
    "00": "Normal (00)",
    "01": "Contextual Anomaly (01)",
    "10": "Point Anomaly (10)",
    "11": "Sensor Fault (11)",
}

MODEL_LABELS = {"svm": "Hybrid SVM-SVR", "rf": "Random Forest", "bilstm": "Bi-Directional LSTM"}
MODEL_COLORS = {"svm": "#3B82F6", "rf": "#10B981", "bilstm": "#F97316"}

try:
    import tensorflow  # noqa: F401

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


# ==========================================================
# CUSTOM CSS — dark navy sidebar, clean white main panel
# ==========================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@400;500;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'DM Sans', sans-serif;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F172A 0%, #1E3A5F 100%);
    }
    section[data-testid="stSidebar"] * {
        color: #E2E8F0 !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        font-family: 'Space Mono', monospace;
    }

    .metric-value {
        font-family: 'Space Mono', monospace;
        font-weight: 700;
    }

    .model-card {
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 10px;
        background: #F8FAFC;
    }
    .model-card-svm   { border-left: 6px solid #3B82F6; }
    .model-card-rf    { border-left: 6px solid #10B981; }
    .model-card-bilstm{ border-left: 6px solid #F97316; }

    .best-model-banner {
        background: linear-gradient(90deg, #1E3A5F 0%, #3B82F6 100%);
        color: white;
        padding: 18px 24px;
        border-radius: 12px;
        margin-bottom: 18px;
    }
    .best-model-banner h2 {
        font-family: 'Space Mono', monospace;
        margin: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================================
# CORE PIPELINE FUNCTIONS
# (mirrors Algo1.ipynb — kept self-contained here per the report's
# single-file Streamlit application design, Section 4.2)
# ==========================================================
def numerical_sort_key(value):
    parts = re.split(r"(\d+)", value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def aggregate_telemetry(uploaded_files):
    sorted_files = sorted(uploaded_files, key=lambda f: numerical_sort_key(f.name))
    frames = [pd.read_csv(io.BytesIO(f.getvalue())) for f in sorted_files]
    df = pd.concat(frames, ignore_index=True)

    try:
        df["Actual_Code"] = df.iloc[:, 5].astype(str) + df.iloc[:, 6].astype(str)
    except IndexError:
        raise ValueError(
            "Expected the Anomaly_Type and Normal columns at positions 5 and 6 "
            "of each CSV — check the uploaded file schema."
        )

    return df, [f.name for f in sorted_files]


def engineer_features(df):
    df = df.copy()
    for col in FEATURES:
        df[f"{col}_roll_mean"] = df[col].rolling(window=ROLL_WINDOW).mean()
        df[f"{col}_roll_std"] = df[col].rolling(window=ROLL_WINDOW).std()
    df = df.bfill().ffill()

    X = df.drop(columns=["Anomaly_Type", "Normal", "Actual_Code"])
    y = df["Actual_Code"].astype(str)
    return df, X, y


def split_and_scale(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = MinMaxScaler()
    scaler.fit(X_train)

    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    X_train_scaled = X_train_scaled.reset_index(drop=True)
    X_test_scaled = X_test_scaled.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_svm_gatekeeper(X_train_scaled, y_train):
    y_binary = (y_train != "00").astype(int)
    ros = RandomOverSampler(random_state=RANDOM_STATE)
    X_res, y_res = ros.fit_resample(X_train_scaled, y_binary)
    svm = SVC(kernel="rbf", C=10.0, gamma="scale", random_state=RANDOM_STATE)
    svm.fit(X_res, y_res)
    return svm


def train_rf_gatekeeper(X_train_scaled, y_train):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_train)
    ros = RandomOverSampler(random_state=RANDOM_STATE)
    X_res, y_res = ros.fit_resample(X_train_scaled, y_encoded)
    rf = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_res, y_res)
    return rf, le


def build_windowed_input(X_scaled_df, lookback=LOOKBACK):
    X_arr = np.asarray(X_scaled_df)
    n = len(X_arr)
    d = X_arr.shape[1] if n else 0
    if n == 0:
        return np.empty((0, lookback, d))
    pad = np.repeat(X_arr[[0]], max(lookback - 1, 0), axis=0)
    padded = np.vstack([pad, X_arr])
    return np.array([padded[i : i + lookback] for i in range(n)])


def train_bilstm_gatekeeper(X_train_scaled, y_train, epochs=50, patience=5):
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import Bidirectional, Dense, Dropout, Input, LSTM
    from tensorflow.keras.models import Sequential

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_train)

    ros = RandomOverSampler(random_state=RANDOM_STATE)
    X_res, y_res = ros.fit_resample(X_train_scaled, y_encoded)
    X_res = pd.DataFrame(X_res, columns=X_train_scaled.columns)

    X_win = build_windowed_input(X_res, lookback=LOOKBACK)
    n_classes = len(le.classes_)

    model = Sequential(
        [
            Input(shape=(LOOKBACK, X_res.shape[1])),
            Bidirectional(LSTM(64)),
            Dropout(0.2),
            Dense(n_classes, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    early_stop = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    history = model.fit(
        X_win, y_res, validation_split=0.15, epochs=epochs, batch_size=64,
        callbacks=[early_stop], verbose=0,
    )
    return model, le, history


def fit_dynamic_bounds(X_train_orig, y_train):
    normal_mask = (y_train == "00").to_numpy()
    X_normal = X_train_orig[normal_mask]

    bounds, min_jumps, fault_thresholds = {}, {}, {}
    for feat in FEATURES:
        mu = X_normal[feat].mean()
        sigma = X_normal[feat].std()
        bounds[feat] = (mu - sigma, mu + sigma)
        min_jumps[feat] = 0.30 * sigma
        fault_thresholds[feat] = max(mu - 2.5 * sigma, 5.0)

    return bounds, min_jumps, fault_thresholds


def fit_svr_triage(X_train_scaled, y_train):
    normal_mask = (y_train == "00").to_numpy()
    X_normal_scaled = X_train_scaled[normal_mask]

    svr_models, residual_thresholds = {}, {}
    for feat in FEATURES:
        X_svr = X_normal_scaled.drop(columns=[feat])
        y_svr = X_normal_scaled[feat]
        svr = SVR(kernel="rbf", C=1.0)
        svr.fit(X_svr, y_svr)
        svr_models[feat] = svr
        residuals = np.abs(y_svr - svr.predict(X_svr))
        residual_thresholds[feat] = np.percentile(residuals, 95)

    return svr_models, residual_thresholds


def svr_categorise(row_scaled, row_orig, bounds, min_jumps, fault_thresholds,
                    svr_models, residual_thresholds):
    if (
        row_orig["Heart_Rate"] < fault_thresholds["Heart_Rate"]
        or row_orig["SpO2"] < fault_thresholds["SpO2"]
    ):
        return "11"

    for feat in ["Heart_Rate", "SpO2", "Body_Temperature"]:
        actual_val = row_orig[feat]
        mean_val = row_orig[f"{feat}_roll_mean"]
        std_val = row_orig[f"{feat}_roll_std"]

        lo, hi = bounds[feat]
        if actual_val < lo or actual_val > hi:
            return "10"

        if pd.notna(std_val):
            allowed_variance = max(3 * std_val, min_jumps[feat])
            if abs(actual_val - mean_val) > allowed_variance:
                return "10"

    context_votes = 0
    for feat in FEATURES:
        input_row = row_scaled.drop(labels=[feat]).to_frame().T
        input_row.columns = [c for c in row_scaled.index if c != feat]
        pred_val = svr_models[feat].predict(input_row)[0]
        residual = abs(row_scaled[feat] - pred_val)
        if residual > residual_thresholds[feat]:
            context_votes += 1

    return "01" if context_votes >= 1 else "10"


def run_inference(X_scaled_df, X_orig_df, y_true, gatekeeper, models, artifacts):
    if gatekeeper == "svm":
        gate_preds = models["svm"].predict(X_scaled_df)
        is_abnormal = gate_preds == 1

    elif gatekeeper == "rf":
        rf, le = models["rf"]
        preds = le.inverse_transform(rf.predict(X_scaled_df))
        is_abnormal = preds != "00"

    elif gatekeeper == "bilstm":
        model, le = models["bilstm"]
        X_win = build_windowed_input(X_scaled_df)
        probs = model.predict(X_win, verbose=0)
        preds = le.inverse_transform(np.argmax(probs, axis=1))
        is_abnormal = preds != "00"

    else:
        raise ValueError(f"Unknown gatekeeper: {gatekeeper!r}")

    y_true_arr = np.asarray(y_true)
    rows = []
    for i in range(len(X_scaled_df)):
        if is_abnormal[i]:
            pred_code = svr_categorise(
                X_scaled_df.iloc[i], X_orig_df.iloc[i],
                artifacts["bounds"], artifacts["min_jumps"], artifacts["fault_thresholds"],
                artifacts["svr_models"], artifacts["residual_thresholds"],
            )
        else:
            pred_code = "00"

        rows.append({
            "Row_ID": int(X_scaled_df.index[i]),
            "Actual": y_true_arr[i],
            "Predicted": pred_code,
            "Match": y_true_arr[i] == pred_code,
        })

    return pd.DataFrame(rows)


def compute_metrics(results_df):
    y_true = results_df["Actual"]
    y_pred = results_df["Predicted"]
    labels = sorted(set(y_true) | set(y_pred))
    target_names = [CLASS_NAMES.get(c, c) for c in labels]

    report_str = classification_report(
        y_true, y_pred, labels=labels, target_names=target_names, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "report_str": report_str,
        "confusion_matrix": cm,
        "labels": labels,
        "target_names": target_names,
    }


@st.cache_data(show_spinner=False)
def load_and_engineer(file_bytes_by_name):
    """Cached data ingestion + feature engineering, keyed on uploaded file
    content so re-uploading identical files skips recomputation
    (Section 4.3.3 of the report)."""
    class _FakeFile:
        def __init__(self, name, data):
            self.name = name
            self._data = data

        def getvalue(self):
            return self._data

    fake_files = [_FakeFile(name, data) for name, data in file_bytes_by_name]
    df, filenames = aggregate_telemetry(fake_files)
    df, X, y = engineer_features(df)
    return df, X, y, filenames


def run_full_pipeline(uploaded_files, selected_models):
    file_bytes_by_name = tuple((f.name, f.getvalue()) for f in uploaded_files)
    df, X, y, filenames = load_and_engineer(file_bytes_by_name)

    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_scale(X, y)

    models = {}
    bilstm_history = None
    rf_model_ref = None

    progress = st.progress(0.0, text="Starting pipeline...")
    total_steps = len(selected_models) + 2  # + Stage 2 setup + inference
    step = 0

    if "svm" in selected_models:
        progress.progress(step / total_steps, text="Training SVM gatekeeper...")
        models["svm"] = train_svm_gatekeeper(X_train_scaled, y_train)
        step += 1

    if "rf" in selected_models:
        progress.progress(step / total_steps, text="Training Random Forest gatekeeper...")
        models["rf"] = train_rf_gatekeeper(X_train_scaled, y_train)
        rf_model_ref = models["rf"][0]
        step += 1

    if "bilstm" in selected_models:
        progress.progress(step / total_steps, text="Training Bi-Directional LSTM gatekeeper...")
        if not TF_AVAILABLE:
            st.warning("TensorFlow is not installed — skipping the BiLSTM gatekeeper.")
        else:
            bilstm_model, bilstm_le, bilstm_history = train_bilstm_gatekeeper(X_train_scaled, y_train)
            models["bilstm"] = (bilstm_model, bilstm_le)
        step += 1

    progress.progress(step / total_steps, text="Fitting Stage 2 SVR triage layer...")
    bounds, min_jumps, fault_thresholds = fit_dynamic_bounds(X_train, y_train)
    svr_models, residual_thresholds = fit_svr_triage(X_train_scaled, y_train)
    artifacts = {
        "bounds": bounds, "min_jumps": min_jumps, "fault_thresholds": fault_thresholds,
        "svr_models": svr_models, "residual_thresholds": residual_thresholds,
    }
    step += 1

    progress.progress(step / total_steps, text="Running inference for all selected models...")
    results, metrics = {}, {}
    for name in models:
        results_df = run_inference(X_test_scaled, X_test, y_test, name, models, artifacts)
        results[name] = results_df
        metrics[name] = compute_metrics(results_df)

    progress.progress(1.0, text="Done.")
    progress.empty()

    return {
        "df": df,
        "results": results,
        "metrics": metrics,
        "rf_model": rf_model_ref,
        "history": bilstm_history,
        "feature_names": list(X_test_scaled.columns),
        "X_test": X_test,
        "y_test": y_test,
        "X_train": X_train,
        "y_train": y_train,
        "total_samples": len(df),
        "test_samples": len(X_test),
        "n_features": X_test_scaled.shape[1],
        "n_classes": y.nunique(),
        "class_dist_train": y_train.value_counts(),
        "class_dist_test": y_test.value_counts(),
    }


# ==========================================================
# SIDEBAR
# ==========================================================
with st.sidebar:
    st.markdown("### \U0001FAC0 WBAN Anomaly Detection")
    st.caption("v2.0 — Expanded Model Suite")
    st.divider()

    st.markdown("#### \U0001F4C1 Upload Telemetry CSVs")
    uploaded_files = st.file_uploader(
        "Upload one or more CSV files",
        type="csv",
        accept_multiple_files=True,
        help="Drag and drop patient telemetry files here. Limit 200MB per file.",
    )

    st.markdown("#### \u2699\ufe0f Model Selection")
    use_svm = st.checkbox("Hybrid SVM-SVR", value=True)
    use_rf = st.checkbox("Random Forest", value=True)
    use_bilstm = st.checkbox(
        "Bi-Directional LSTM",
        value=TF_AVAILABLE,
        disabled=not TF_AVAILABLE,
        help=None if TF_AVAILABLE else "TensorFlow is not installed in this environment.",
    )

    run_clicked = st.button("\U0001F680 Run Pipeline", use_container_width=True, type="primary")

    st.divider()
    st.caption("Rolling window: 5 steps")
    st.caption("Test split: 20% stratified")
    st.caption("Oversampler: RandomOverSampler")


# ==========================================================
# MAIN PANEL
# ==========================================================
st.markdown(
    """
    <h1 style="font-family:'Space Mono',monospace;">WBAN Anomaly Detection Dashboard</h1>
    <p style="color:#64748B;">Two-Stage Hybrid SVM-SVR · Random Forest · Bi-Directional LSTM · Comparative Evaluation</p>
    """,
    unsafe_allow_html=True,
)

selected_models = [name for name, flag in [("svm", use_svm), ("rf", use_rf), ("bilstm", use_bilstm)] if flag]

if run_clicked:
    if not uploaded_files:
        st.error("Please upload at least one telemetry CSV before running the pipeline.")
    elif not selected_models:
        st.error("Please select at least one gatekeeper model.")
    else:
        with st.spinner("Running the two-stage pipeline..."):
            st.session_state["pipeline_results"] = run_full_pipeline(uploaded_files, selected_models)

pipeline_results = st.session_state.get("pipeline_results")

if pipeline_results is None:
    st.info("Upload telemetry CSVs, select your gatekeeper model(s), and click **Run Pipeline** to begin.")
    st.stop()

# ---- Summary metric cards ----
st.markdown("#### \U0001F4CB Pipeline Overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Samples", f"{pipeline_results['total_samples']:,}")
c2.metric("Test Samples", f"{pipeline_results['test_samples']:,}")
c3.metric("Feature Columns", pipeline_results["n_features"])
c4.metric("Classes", pipeline_results["n_classes"])

# ---- Best model banner ----
metrics = pipeline_results["metrics"]
if metrics:
    best_name = max(metrics, key=lambda n: metrics[n]["accuracy"])
    best = metrics[best_name]
    st.markdown(
        f"""
        <div class="best-model-banner">
            <div style="font-size:0.85rem; opacity:0.85;">\U0001F3C6 BEST PERFORMING MODEL</div>
            <h2>{MODEL_LABELS[best_name]}</h2>
            <div style="font-family:'Space Mono',monospace;">
                Accuracy {best['accuracy']*100:.2f}% · F1 {best['f1_macro']:.4f} · Precision {best['precision_macro']:.4f}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---- Class distribution warning ----
missing_test_classes = set(pipeline_results["class_dist_train"].index) - set(pipeline_results["class_dist_test"].index)
if missing_test_classes:
    pretty = ", ".join(CLASS_NAMES.get(c, c) for c in sorted(missing_test_classes))
    st.warning(
        f"The test split has zero examples of: **{pretty}**. Their precision/recall/F1 will "
        f"show as 0 in the reports below, which drags down the macro-averaged scores — this "
        f"reflects the uploaded dataset, not a modelling error."
    )

st.divider()

# ==========================================================
# TABS — one per model + Comparison
# ==========================================================
tab_labels = [MODEL_LABELS[n] for n in pipeline_results["results"]] + ["\U0001F4CA Comparison"]
tabs = st.tabs(tab_labels)

model_keys = list(pipeline_results["results"].keys())

for i, name in enumerate(model_keys):
    with tabs[i]:
        m = metrics[name]
        results_df = pipeline_results["results"][name]

        mc1, mc2, mc3 = st.columns(3)
        mc1.markdown(
            f'<div class="model-card model-card-{name}"><b>Accuracy</b><br>'
            f'<span class="metric-value">{m["accuracy"]*100:.2f}%</span></div>',
            unsafe_allow_html=True,
        )
        mc2.markdown(
            f'<div class="model-card model-card-{name}"><b>F1 (macro)</b><br>'
            f'<span class="metric-value">{m["f1_macro"]:.4f}</span></div>',
            unsafe_allow_html=True,
        )
        mc3.markdown(
            f'<div class="model-card model-card-{name}"><b>Precision (macro)</b><br>'
            f'<span class="metric-value">{m["precision_macro"]:.4f}</span></div>',
            unsafe_allow_html=True,
        )

        with st.expander("Full classification report", expanded=False):
            st.code(m["report_str"])

        st.markdown("##### Confusion Matrix & Class Distribution")
        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(5, 4.2))
            sns.heatmap(
                m["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
                xticklabels=m["labels"], yticklabels=m["labels"], ax=ax,
            )
            ax.set_title(f"Confusion Matrix — {MODEL_LABELS[name]}")
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with col2:
            dist_df = pd.DataFrame({
                "Actual": results_df["Actual"].value_counts(),
                "Predicted": results_df["Predicted"].value_counts(),
            }).fillna(0).reindex(sorted(set(results_df["Actual"]) | set(results_df["Predicted"])))
            fig, ax = plt.subplots(figsize=(5, 4.2))
            dist_df.plot(kind="bar", ax=ax, color=["#3B82F6", "#F59E0B"])
            ax.set_title(f"Class Distribution — {MODEL_LABELS[name]}")
            ax.tick_params(axis="x", rotation=0)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        # Model-specific extras
        if name == "bilstm" and pipeline_results["history"] is not None:
            st.markdown("##### BiLSTM Training History")
            history = pipeline_results["history"]
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(5, 3.5))
                ax.plot(history.history["accuracy"], label="Train")
                ax.plot(history.history["val_accuracy"], label="Val", linestyle="--")
                ax.set_title("BiLSTM — Accuracy")
                ax.legend()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            with col2:
                fig, ax = plt.subplots(figsize=(5, 3.5))
                ax.plot(history.history["loss"], label="Train")
                ax.plot(history.history["val_loss"], label="Val", linestyle="--")
                ax.set_title("BiLSTM — Loss")
                ax.legend()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

        if name == "rf" and pipeline_results["rf_model"] is not None:
            st.markdown("##### Random Forest — Top Feature Importances")
            importances = pd.Series(
                pipeline_results["rf_model"].feature_importances_,
                index=pipeline_results["feature_names"],
            ).sort_values(ascending=True).tail(15)
            fig, ax = plt.subplots(figsize=(9, 5))
            importances.plot(kind="barh", ax=ax, color="#10B981")
            ax.set_title("Random Forest — Top Feature Importances")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        st.markdown("##### Prediction Samples")
        st.dataframe(results_df.head(200), use_container_width=True, height=300)

        st.download_button(
            f"\U0001F4E5 Download {MODEL_LABELS[name]} Predictions (CSV)",
            data=results_df.to_csv(index=False).encode("utf-8"),
            file_name=f"predictions_{name}.csv",
            mime="text/csv",
        )

# ---- Comparison tab ----
with tabs[-1]:
    st.markdown("##### \U0001F4CA Comparative Performance")

    comparison_df = pd.DataFrame(
        [
            {
                "Model": MODEL_LABELS[name],
                "Accuracy (%)": round(metrics[name]["accuracy"] * 100, 2),
                "F1 (macro)": round(metrics[name]["f1_macro"], 4),
                "Precision (macro)": round(metrics[name]["precision_macro"], 4),
            }
            for name in model_keys
        ]
    ).sort_values("Accuracy (%)", ascending=False).reset_index(drop=True)
    comparison_df.insert(0, "Rank", range(1, len(comparison_df) + 1))

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        colors = [MODEL_COLORS[n] for n in model_keys]
        bars = ax.bar(comparison_df["Model"], comparison_df["Accuracy (%)"], color=colors[: len(comparison_df)])
        ax.set_title("Detection Accuracy — All Models")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 100)
        for bar, val in zip(bars, comparison_df["Accuracy (%)"]):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 1, f"{val:.2f}%", ha="center")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        x = np.arange(len(comparison_df))
        width = 0.35
        ax.bar(x - width / 2, comparison_df["F1 (macro)"], width, label="F1 (macro)", color="#3B82F6")
        ax.bar(x + width / 2, comparison_df["Precision (macro)"], width, label="Precision (macro)", color="#93C5FD")
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df["Model"])
        ax.set_title("Macro F1-Score & Precision — All Models")
        ax.set_ylim(0, 1.05)
        ax.legend()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("###### Performance Table")
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    st.caption(
        "Note: the project report's Table 5 also lists a 'Static Threshold (baseline)' row "
        "(~62% accuracy). That baseline is not computed by this dashboard, so it is omitted "
        "here rather than shown as an unverified number."
    )

    st.download_button(
        "\U0001F4E5 Download Comparison CSV",
        data=comparison_df.to_csv(index=False).encode("utf-8"),
        file_name="model_comparison.csv",
        mime="text/csv",
    )

# ---- Cache pipeline results in session state (Section 4.3.3 pattern) ----
st.session_state["pipeline_results"] = pipeline_results
