# WBAN Anomaly Detection — Dynamic Two-Stage Hybrid ML Framework

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

A dynamic hybrid machine learning system for anomaly detection and classification of health signals in Wireless Body Area Networks (WBANs). The framework combines a fast gatekeeper screening layer with an SVR-based contextual triage layer to reduce false alarms and categorize physiological anomalies into clinically meaningful classes — deployed through an interactive Streamlit dashboard.

> End-Sem Laboratory Project (CS F366) — BITS Pilani, Hyderabad Campus
> Author: Harshal Shah (2023A7PS0055H)
> Guide: Prof. Chittaranjan Hota

---

## Overview

Static thresholding — the standard approach in deployed WBAN systems — is population-level, temporally blind, and unable to distinguish sensor faults from genuine clinical events. This leads to alarm fatigue and missed contextual deterioration.

This project implements a **two-stage hybrid pipeline**:

1. **Stage 1 — Gatekeeper Classification:** A fast screening layer flags observations as normal or abnormal. Three interchangeable gatekeeper models are supported:
   - Support Vector Machine (RBF kernel)
   - Random Forest (balanced class weighting)
   - Bi-Directional LSTM (temporal windowed sequences)

2. **Stage 2 — SVR-Based Triage:** Flagged observations are categorized using patient-adaptive dynamic thresholds and SVR residual scoring into:

   | Code | Class | Description |
   |------|-------|-------------|
   | `00` | Normal | All readings within expected bounds |
   | `01` | Contextual Anomaly | Normal in isolation, anomalous given patient history |
   | `10` | Point Anomaly | Isolated extreme value, likely an artefact |
   | `11` | Sensor Fault | Physiologically implausible reading (hardware failure) |

The complete pipeline is exposed through a Streamlit dashboard that accepts raw telemetry CSVs, runs training/inference in the background, and renders comparative performance visualizations.

---

## Features

- Multi-file CSV telemetry ingestion with chronological sorting and concatenation
- Rolling-window temporal feature engineering (mean & std over a 5-step window)
- Patient-adaptive dynamic thresholding for contextual anomaly detection
- Three swappable gatekeeper models with a unified inference interface
- Interactive Streamlit dashboard with:
  - Per-model result tabs (accuracy, F1, precision, classification report)
  - Confusion matrix and class distribution visualizations
  - BiLSTM training history and Random Forest feature importance plots
  - Interactive time-series anomaly overlay
  - CSV export of predictions

---

## Project Structure

```
WBAN-Anomaly-Detection/
├── wban_dashboard.py       # Main Streamlit application (data pipeline + models + UI)
├── requirements.txt        # Pinned dependency list
├── notebooks/
│   └── Algo1.ipynb         # Colab-compatible notebook for batch experimentation
├── data/
│   └── dataset.zip         # Sample multi-patient telemetry CSVs (extract before use)
└── docs/
    └── paper.pdf            # Project report
```

---

## Tech Stack

| Library | Version | Role |
|---|---|---|
| scikit-learn | ≥ 1.4 | SVM, Random Forest, SVR, metrics, scaling |
| TensorFlow / Keras | ≥ 2.16 | BiLSTM architecture and training |
| imbalanced-learn | ≥ 0.12 | RandomOverSampler |
| pandas | ≥ 2.2 | Data ingestion, feature frames |
| NumPy | ≥ 1.26 | Numerical operations, windowing |
| Matplotlib | ≥ 3.8 | Visualization |
| Seaborn | ≥ 0.13 | Visualization |
| Streamlit | ≥ 1.35 | Interactive dashboard |

---

## Getting Started

### Prerequisites
- Python 3.10+

### Installation

```bash
git clone https://github.com/HarshalShah0508/WBAN-Anomaly-Detection.git
cd WBAN-Anomaly-Detection
pip install -r requirements.txt
```

### Running the Dashboard

```bash
streamlit run wban_dashboard.py
```

Then open the local URL Streamlit prints (typically `http://localhost:8501`), upload one or more telemetry CSVs (unzip `data/dataset.zip` for a sample multi-patient dataset), select the gatekeeper model(s) to run, and click **Run Pipeline**.

### Running the Notebook

`notebooks/Algo1.ipynb` is Colab-compatible and can be used for batch experimentation, model tuning, and generating the comparative figures independently of the dashboard.

---

## Methodology

### Data Pipeline
1. **Sensor modalities:** Body Temperature, Heart Rate, Pulse Rate, SpO₂, ECG
2. **Temporal feature engineering:** Rolling mean and standard deviation over a 5-step window (expands 5 raw features → 15 features)
3. **Normalization:** Min-Max scaling fit only on training data
4. **Class balancing:** Stratified 80/20 train-test split, followed by `RandomOverSampler` on the training set

### Stage 1 — Gatekeepers
- **SVM:** RBF kernel, `C=10.0`, `gamma='scale'`, binary output
- **Random Forest:** 100 trees, `class_weight='balanced'`, direct multiclass prediction
- **BiLSTM:** 5-step lookback window, 64-unit bidirectional layer, dropout 0.2, softmax over 4 classes, trained with early stopping (patience=5)

### Stage 2 — Triage Logic
Applied in strict priority order to flagged observations:
1. **Sensor fault check** — implausible HR/SpO₂ values
2. **Point anomaly check** — deviation from patient-adaptive bounds or rolling mean
3. **Contextual anomaly check** — SVR residual exceeds the 95th percentile of training residuals

---

## Results

Evaluated on a multi-patient telemetry dataset (held-out test set):

| Model | Accuracy | F1 (Macro) | Precision (Macro) |
|---|---|---|---|
| Hybrid SVM–SVR | ~87% | ~0.84 | ~0.83 |
| Random Forest | ~85.5% | ~0.83 | ~0.82 |
| Bi-Directional LSTM | see note below | see note below | see note below |
| Static Threshold (baseline) | ~62% | ~0.51 | ~0.49 |

> **Note:** There is a discrepancy between the summary table and the dashboard-generated comparison chart/confusion matrix regarding BiLSTM performance. Recommend re-verifying against the notebook's actual output before finalizing these numbers.

Key findings:
- Rolling standard deviation features are the strongest discriminators for the Random Forest model (see feature importance plot).
- Patient-adaptive dynamic thresholds correctly avoid false-flagging naturally elevated baselines while catching gradual physiological drift that fixed thresholds miss.

---

## Future Work

- Edge deployment on gateway hardware (Raspberry Pi / Jetson Nano) with TensorFlow Lite quantization
- Federated learning for privacy-preserving multi-site training
- Online adaptive retraining with concept-drift detection (ADWIN / Page-Hinkley)
- Multi-modal sensor fusion (GSR, accelerometry, ambient sensors) via attention/Transformer architectures
- HL7-FHIR integration for direct EHR alerting
- SHAP-based explainability for per-observation anomaly attribution

---

## Acknowledgements

Developed under the guidance of Prof. Chittaranjan Hota, Department of Computer Science & Information Systems, BITS Pilani, Hyderabad Campus.

## License

MIT — see [LICENSE](LICENSE).
