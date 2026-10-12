# AI-Enabled Micro-channel Battery Thermal Management System (FreeTwinEV)

This repository contains the complete research, machine learning forecasting pipeline, decision-support layer, and interactive dashboard for the **AI-Enabled Micro-channel Battery Thermal Management System** using the **FreeTwinEV P45B 1s4p Battery Module Measurement** dataset.

---

## 📌 Project Overview

- **Dataset**: FreeTwinEV P45B 1S4P Battery Module (4 Molicel INR21700-P45B cells in a liquid-cooled module with 25 temperature channels).
- **Core Objective**: Predict future maximum cell-surface temperature change ($\Delta T_{\max}$) at both **60-second** and **300-second** forecasting horizons, reconstruct absolute temperatures, evaluate risk classifications, and provide actionable cooling recommendations.
- **Architecture**:
  - **Forecasting Models**: Ridge Regression ($\alpha = 1.0$) for 60s horizon, Linear Regression (OLS) for 300s horizon using extended 21 features.
  - **Risk Layer**: Dynamic thresholding and rule-based risk mitigation decision support.
  - **Interactive Dashboard**: Modern web interface and Streamlit application for real-time monitoring and playback.

---

## 📁 Repository Structure

```
├── dashboard/               # Frontend application (React / Vite dashboard)
├── data/                    # Dataset directory (FreeTwinEV laboratory measurements)
├── notebooks/               # Step-by-step Jupyter notebooks (Phases 0-11)
│   ├── 01_eda.ipynb ... 16_phase10_dashboard.ipynb
├── reports/                 # Comprehensive technical reports & phase documentation
├── results/                 # Evaluation metrics, preprocessed datasets & generated figures
├── src/                     # Core Python source code & pipeline scripts
├── .gitignore               # Git ignore patterns
└── README.md                # Project documentation
```

---

## 🚀 Getting Started

### 1. Python Environment Setup
```bash
pip install -r requirements.txt # or install numpy, pandas, scikit-learn, streamlit, plotly
```

### 2. Running Data Processing & Model Pipeline
```bash
# Run baseline models
python src/run_freetwinev_baselines.py

# Run Phase 5 - 9 pipelines
python src/run_freetwinev_phase5.py
python src/run_freetwinev_phase6.py
python src/run_freetwinev_phase7.py
python src/run_freetwinev_phase8.py
python src/run_freetwinev_phase9.py
```

### 3. Launching the Dashboard
```bash
cd dashboard
npm install
npm run dev
```

---

## 📊 Evaluation Summary

| Horizon | Selected Model | Features | Val MAE (°C) | Val RMSE (°C) | Test MAE (°C) | Test RMSE (°C) | Test $R^2$ |
|---|---|---|:---:|:---:|:---:|:---:|:---:|
| **60s** | Ridge Regression ($\alpha=1.0$) | Extended 21 | 0.1598 | 0.7228 | 0.2271 | 0.8256 | 0.8060 |
| **300s** | Linear Regression (OLS) | Extended 21 | 2.2582 | 4.5396 | 1.9302 | 4.9649 | 0.5836 |

---

## 📄 License & Attribution
Data sourced from FreeTwinEV laboratory measurement dataset under CC BY 4.0.
