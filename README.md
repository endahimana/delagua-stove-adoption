# 🇷🇼 DelAgua Stove Adoption Analytics
### Predictive Decision Support System for Clean Cooking Programs

**Live Application:** [Insert Your Streamlit Cloud Link Here]

## 📋 Project Overview
This project was developed as part of the DelAgua Data Scientist assessment. It provides a machine-learning-driven framework to identify households at risk of low stove adoption (defined as <30% reduction in traditional fuel usage). 

By identifying these households early, program managers can deploy field teams proactively, ensuring the sustainability of clean cooking habits and the integrity of carbon credit generation.

## 🚀 Key Features
- **Executive Dashboard:** High-level program KPIs and interactive spatial risk mapping using OpenStreetMap.
- **Predictive Oracle:** A "What-If" tool for field officers to estimate adoption probability for new households.
- **Batch Intelligence:** Process entire village distribution lists via CSV upload to generate prioritized follow-up sheets.
- **Data Integrity Pipeline:** Automated detection and correction of geographic coordinate swaps and standardized district nomenclature.

## 🧠 The Model
- **Algorithm:** `HistGradientBoostingClassifier`
- **Key Metric:** ROC-AUC: **0.6364**
- **Core Insight:** The analysis revealed the **Market Climb Index** (interaction of distance to market and elevation) as the primary driver of adoption risk, outperforming traditional demographic indicators.

## 🛠️ Tech Stack
- **Language:** Python 3.11+
- **Framework:** Streamlit
- **ML Libraries:** Scikit-Learn, Joblib
- **Data & Viz:** Pandas, Numpy, Plotly, Seaborn

## 📂 Directory Structure
```text
delagua_stove_app/
├── app.py                # Main Streamlit application
├── utils.py              # Shared cleaning & feature engineering logic
├── train_model.py        # Automated model training pipeline
├── requirements.txt      # Deployment dependencies
├── data/
│   └── processed_stove_data.csv
└── models/
    ├── stove_adoption_model.pkl
    └── model_metadata.json