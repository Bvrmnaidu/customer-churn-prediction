# LinkedIn Post — Customer Churn Prediction Project

---

## POST (copy-paste ready)

I built an end-to-end Customer Churn Prediction system — from raw data to a live interactive dashboard. Here's what I learned 👇

📊 The business problem:
Acquiring a new customer costs 5–7x more than keeping one. Churn prediction isn't just an ML exercise — it's a direct revenue problem.

🏗️ What I built:
→ Synthetic telecom dataset: 5,000 customers, 19 features, 38% churn rate
→ Feature engineering: 8 new signals (avg monthly spend, service count, tenure segments)
→ 4 models compared: Logistic Regression, Random Forest, Gradient Boosting, Decision Tree
→ Best model selected by AUC-ROC (most appropriate for imbalanced churn data)
→ 4-page Streamlit dashboard: Overview, Data Explorer, Model Comparison, Real-Time Predictor

🔍 Key finding that surprised me:
Month-to-month customers churn at 3x the rate of two-year contract holders. But the real insight? The first 12 months are critical — churn risk drops sharply after the 1-year mark. This has direct implications for onboarding strategy.

🛠 Tech stack:
Python · Scikit-learn · Pandas · Streamlit · Feature Engineering · ML Pipeline

💻 Full code + README on GitHub: [LINK]
🎯 Live demo: [STREAMLIT LINK]

What retention strategy would you implement for a high-risk customer on a month-to-month contract? 👇

#DataScience #MachineLearning #Python #CustomerChurn #MLPipeline #Streamlit #OpenToWork

---

## MEDIUM ARTICLE OUTLINE

**Title:** "Building a Production-Grade Customer Churn Predictor: From Raw Data to Live Dashboard"

**Subtitle:** "An end-to-end walkthrough covering data engineering, feature design, model comparison, and Streamlit deployment"

---

### Introduction (~150 words)
- Why churn prediction matters (revenue impact, cost of acquisition)
- What this article covers
- Link to live demo and GitHub

### Section 1: Understanding the Problem (~200 words)
- Framing churn as a binary classification with class imbalance
- Why AUC-ROC matters more than accuracy for this problem
- Dataset overview — features and target

### Section 2: Feature Engineering (~300 words)
- Why raw features are never enough
- Walk through each engineered feature with rationale
- Code snippet: the engineer_features() function
- Highlight: why avg_monthly_spend is more informative than total_charges alone

### Section 3: The ML Pipeline (~300 words)
- sklearn Pipeline + ColumnTransformer — why this prevents data leakage
- Preprocessing choices: StandardScaler for numerics, OneHotEncoder for categoricals
- Model selection rationale
- Results table with code snippet

### Section 4: Key Findings (~250 words)
- 3 most important insights from the data
- Feature importance chart explanation
- Business interpretation of each finding

### Section 5: The Streamlit Dashboard (~200 words)
- 4-page structure walkthrough
- Real-time prediction logic
- Risk segmentation: High / Medium / Low with retention actions

### Conclusion (~100 words)
- What you'd do with more time (SHAP explainability, real dataset, retraining pipeline)
- GitHub link, LinkedIn, call to action

---

**Estimated read time:** 8 minutes
**Target publication:** Towards Data Science or Medium Data Science tag
**Tags:** Data Science, Machine Learning, Python, Streamlit, Customer Churn

---

## DEPLOYMENT STEPS (for Streamlit Cloud)

1. Push project to GitHub
2. Go to share.streamlit.io
3. Connect your GitHub repo
4. Set main file path: app/streamlit_app.py
5. Add requirements.txt — Streamlit Cloud reads it automatically
6. Copy the public URL and add to LinkedIn Featured section + GitHub README
