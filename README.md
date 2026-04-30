# 📊 Customer Churn Prediction — End-to-End ML Pipeline

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange?logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?logo=streamlit&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-IBM%20Telco-blue)
![AUC](https://img.shields.io/badge/Best%20AUC-0.84-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

> **🚀 Live Demo →** [your-app.streamlit.app](https://share.streamlit.io)

Predict which telecom customers are at risk of churning — using the real IBM Telco dataset, a production-grade sklearn pipeline, permutation-based explainability, and a 5-page interactive Streamlit dashboard.

---

## 🎯 Business Problem

Customer churn costs telecom companies millions annually. Acquiring a new customer costs **5–7× more** than retaining one. This project answers three questions:

1. **Who** is likely to churn?
2. **Why** does the model predict churn for that customer?
3. **What action** should we take?

---

## 📊 Dataset

**IBM Telco Customer Churn** — 7,032 real customers, 20 features.

| Feature | Type | Description |
|---|---|---|
| `tenure` | Numeric | Months as a customer |
| `monthlycharges` | Numeric | Monthly bill ($) |
| `contract` | Categorical | Month-to-month / 1yr / 2yr |
| `internetservice` | Categorical | DSL / Fiber optic / None |
| `paymentmethod` | Categorical | Electronic check / Auto-pay / etc. |
| `onlinesecurity` | Binary | Security add-on |
| `techsupport` | Binary | Support add-on |
| ... | ... | 13 more features |
| `churn` | **Target** | Yes / No → mapped to 1 / 0 |

**Churn rate: 26.6%** — class imbalance handled via AUC-ROC evaluation.

---

## ⚙️ ML Pipeline

### Feature Engineering

10 new features engineered from domain knowledge:

| Feature | Formula | Rationale |
|---|---|---|
| `avg_monthly_spend` | `totalcharges / (tenure + 1)` | True spending rate smoothed by time |
| `service_count` | Sum of 6 add-on services | Switching cost proxy |
| `charge_per_service` | `monthlycharges / (service_count + 1)` | Perceived value signal |
| `tenure_x_charges` | `tenure × monthlycharges` | Loyalty × spend interaction |
| `new_customer` | `tenure < 6` | Early churn risk flag |
| `long_tenure` | `tenure > 24` | Loyalty signal |
| `auto_payment` | Bank transfer or credit card | Friction reduction signal |
| `high_value_customer` | Charges above median | Revenue prioritisation |

### Preprocessing (no data leakage)

```python
ColumnTransformer([
    ("num", Pipeline([MedianImputer, StandardScaler]), num_cols),
    ("cat", Pipeline([ModeImputer, OneHotEncoder]),    cat_cols),
])
```

Built with `sklearn.Pipeline` — fit only on training data, applied to test set.

### Model Comparison

| Model | AUC-ROC | F1 | Accuracy |
|---|---|---|---|
| **Logistic Regression** | **0.8386** | **0.5818** | **79.5%** |
| Gradient Boosting | 0.8352 | 0.5739 | 78.9% |
| Random Forest | 0.8324 | 0.5634 | 78.9% |
| Decision Tree | 0.8148 | 0.5650 | 79.3% |

**Evaluation metric: AUC-ROC** — correct choice for imbalanced binary classification.

### Explainability

- **Permutation Importance** — measures accuracy drop when each feature is shuffled
- **SHAP values** — install `shap` for per-prediction waterfall plots (see upgrade note)

---

## 🖥️ Dashboard (5 pages)

| Page | Content |
|---|---|
| 📈 **Overview** | KPI cards, churn by contract/internet/tenure/payment, revenue at risk |
| 💡 **Business Insights** | Revenue impact, loyalty analysis, service bundling, top 5 recommendations |
| 🤖 **Model Performance** | Model comparison table, AUC/F1 charts, confusion matrix with cost interpretation |
| 🔎 **Explainability** | Permutation importance chart, feature-by-feature business explanation |
| 🎯 **Predict Customer** | Real-time prediction with risk factors, why-this-score explanation, retention actions |

---

## 🔑 Key Findings

| Finding | Implication |
|---|---|
| Month-to-month customers churn at **42%** vs 11% for two-year contracts | Converting to annual contracts is the #1 retention lever |
| Fiber optic customers churn at **~41%** despite premium pricing | Value gap — security/support bundles reduce this |
| **First 6 months** have the highest churn rate by far | Strong onboarding programme is critical |
| Electronic check users churn at **~45%** vs 15% for auto-pay | Incentivise auto-payment switch |
| Customers with **3+ add-ons** churn significantly less | Bundle strategy increases switching cost |
| Churned customers pay **more** on average than retained ones | High-value customers are highest risk — prioritise proactively |

---

## 🚀 Running Locally

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/customer-churn-prediction.git
cd customer-churn-prediction

# 2. Install
pip install -r requirements.txt

# 3. Add dataset
# Place WA_Fn-UseC_-Telco-Customer-Churn.csv into data/

# 4. Train
python src/train_pipeline.py

# 5. Launch
streamlit run app/streamlit_app.py
```

## ☁️ Deploy on Streamlit Cloud (free)

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → set main file: `app/streamlit_app.py`
4. Add `WA_Fn-UseC_-Telco-Customer-Churn.csv` to `data/` folder
5. Live in ~2 minutes — copy URL for LinkedIn & README

---

## 📁 Project Structure

```
customer-churn-prediction/
├── src/
│   └── train_pipeline.py      # Full ML pipeline (load → engineer → train → evaluate → save)
├── app/
│   └── streamlit_app.py       # 5-page interactive dashboard
├── models/
│   ├── best_model.pkl         # Trained best model
│   ├── feature_config.pkl     # Feature column names
│   └── results.json           # All metrics + business stats + feature importance
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Data & ML | Python, Pandas, NumPy, Scikit-learn |
| Explainability | Permutation Importance, (SHAP ready) |
| Dashboard | Streamlit |
| Deployment | Streamlit Cloud |
| Version Control | Git / GitHub |

---

## ⬆️ Upgrade Path (Next Steps)

- [ ] Add SHAP waterfall + beeswarm plots (`pip install shap`)
- [ ] Add retraining pipeline with new data
- [ ] Dockerise for production deployment
- [ ] Add threshold tuning (optimise for recall to catch more churners)
- [ ] Add A/B test simulation for retention strategy impact

---

## 👤 Author

**Arjun Naidu** · M.Sc. Applied Data Science & Analytics · SRH University Hamburg  
[LinkedIn](https://www.linkedin.com/in/arjunnaidu7013) · [GitHub](https://github.com/YOUR_USERNAME) · Hamburg, Germany

---

*⭐ Star this repo if you found it useful — helps others find it!*
