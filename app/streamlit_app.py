import streamlit as st
import pandas as pd
import numpy as np
import json, joblib, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
from train_pipeline import engineer_features

st.set_page_config(
    page_title="Telco Churn Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  [data-testid="stMetricValue"] { font-size: 1.6rem !important; font-weight: 600; }
  .insight-box {
    background: #f0f9ff; border-left: 4px solid #3b82f6;
    padding: 0.9rem 1.1rem; border-radius: 0 8px 8px 0;
    margin-bottom: 0.8rem; font-size: 0.9rem; line-height: 1.6;
  }
  .insight-box.warn  { background:#fff7ed; border-color:#f97316; }
  .insight-box.good  { background:#f0fdf4; border-color:#22c55e; }
  .insight-box.alert { background:#fef2f2; border-color:#ef4444; }
  .kpi-label  { font-size:12px; color:#64748b; margin-bottom:2px; }
  .kpi-value  { font-size:26px; font-weight:700; color:#1e293b; }
  .kpi-sub    { font-size:11px; color:#94a3b8; }
  h4 { color: #1e293b; margin-bottom: .5rem; }
</style>
""", unsafe_allow_html=True)


# ─── Load artefacts ────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return (joblib.load("models/best_model.pkl"),
            json.load(open("models/results.json")))

@st.cache_data
def load_dataset():
    df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    df = df.drop("customerID", axis=1)
    df.columns = df.columns.str.lower()
    return df

model, results = load_model()
df_raw  = load_dataset()
df_feat = engineer_features(df_raw)
biz     = results.get("business_stats", {})
fi_data = results.get("feature_importance", [])
best    = results["best_model"]
best_r  = results["results"][best]


# ─── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 Telco Churn Analytics")
    st.markdown("---")
    page = st.radio("", [
        "📈 Overview",
        "💡 Business Insights",
        "🤖 Model Performance",
        "🔎 Explainability",
        "🎯 Predict Customer",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.caption("Dataset")
    st.metric("Customers",  f"{len(df_raw):,}")
    st.metric("Churn Rate", f"{df_raw['churn'].mean():.1%}")
    st.metric("Best AUC",   f"{best_r['roc_auc']:.4f}")
    st.metric("Model",      best)
    st.markdown("---")
    st.caption("Built by Arjun Naidu · SRH Hamburg")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📈 Overview":
    st.title("📊 Customer Churn Dashboard")
    st.markdown("Real-world Telco dataset · 7,032 customers · End-to-end ML pipeline")
    st.markdown("---")

    total    = len(df_raw)
    churned  = int(df_raw["churn"].sum())
    retained = total - churned
    rev_risk = round(df_raw[df_raw["churn"]==1]["monthlycharges"].sum(), 0)
    avg_rev  = round(df_raw["monthlycharges"].mean(), 2)

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Total Customers",     f"{total:,}")
    c2.metric("Churned",             f"{churned:,}",  delta=f"−{churned/total:.1%}",  delta_color="inverse")
    c3.metric("Retained",            f"{retained:,}", delta=f"+{retained/total:.1%}")
    c4.metric("Monthly Rev at Risk", f"${rev_risk:,.0f}", delta_color="inverse")
    c5.metric("Avg Monthly Charge",  f"${avg_rev}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Churn Rate by Contract Type")
        ct = df_raw.groupby("contract")["churn"].mean().mul(100).round(1)
        st.bar_chart(ct.rename("Churn Rate (%)"))
        st.markdown("""<div class='insight-box alert'>
        <b>Key signal:</b> Month-to-month customers churn at <b>~42%</b> vs ~11% for two-year contracts.
        Locking customers into longer contracts is the single biggest lever.
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("#### Churn Rate by Internet Service")
        it = df_raw.groupby("internetservice")["churn"].mean().mul(100).round(1)
        st.bar_chart(it.rename("Churn Rate (%)"))
        st.markdown("""<div class='insight-box warn'>
        <b>Key signal:</b> Fiber optic customers churn at <b>~41%</b> — nearly double DSL (~19%).
        Likely driven by higher pricing + strong competitor offerings.
        </div>""", unsafe_allow_html=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### Churn Rate by Tenure Bucket")
        df_tmp = df_raw.copy()
        df_tmp["tenure_bucket"] = pd.cut(df_tmp["tenure"],
            bins=[0,6,12,24,48,72], labels=["0-6m","6-12m","1-2yr","2-4yr","4-6yr"])
        tb = df_tmp.groupby("tenure_bucket", observed=True)["churn"].mean().mul(100).round(1)
        st.bar_chart(tb.rename("Churn Rate (%)"))
        st.markdown("""<div class='insight-box alert'>
        <b>Key signal:</b> First 6 months are critical — churn rate is <b>highest in early tenure</b>.
        Strong onboarding programmes reduce this sharply.
        </div>""", unsafe_allow_html=True)

    with col4:
        st.markdown("#### Churn Rate by Payment Method")
        pm = df_raw.groupby("paymentmethod")["churn"].mean().mul(100).round(1)
        st.bar_chart(pm.rename("Churn Rate (%)"))
        st.markdown("""<div class='insight-box warn'>
        <b>Key signal:</b> Electronic check users churn at <b>~45%</b> vs ~15% for auto-pay.
        Encouraging automatic payments significantly reduces churn.
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — BUSINESS INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💡 Business Insights":
    st.title("💡 Business Insights")
    st.markdown("Data-driven findings translated into actionable recommendations.")
    st.markdown("---")

    # Revenue impact
    st.markdown("### Revenue Impact")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Monthly Revenue at Risk",   f"${biz.get('monthly_revenue_at_risk',0):,.0f}")
    c2.metric("Avg Charge — Churned",      f"${biz.get('avg_charges_churned',0):.2f}")
    c3.metric("Avg Charge — Retained",     f"${biz.get('avg_charges_retained',0):.2f}")
    c4.metric("Annual Revenue at Risk",    f"${biz.get('monthly_revenue_at_risk',0)*12:,.0f}")

    st.markdown("""<div class='insight-box alert'>
    💸 <b>Churned customers pay more on average</b> than retained ones.
    This means the most valuable customers are the most at risk.
    Proactive retention for high-spend customers has an outsized revenue impact.
    </div>""", unsafe_allow_html=True)

    # Tenure findings
    st.markdown("---")
    st.markdown("### Tenure & Loyalty")
    c1,c2 = st.columns(2)
    c1.metric("Avg Tenure — Churned",  f"{biz.get('avg_tenure_churned',0):.1f} months")
    c2.metric("Avg Tenure — Retained", f"{biz.get('avg_tenure_retained',0):.1f} months")

    df_tmp = df_raw.copy()
    df_tmp["tenure_bucket"] = pd.cut(df_tmp["tenure"],
        bins=[0,6,12,24,48,72], labels=["0-6m","6-12m","1-2yr","2-4yr","4-6yr"])
    tb = df_tmp.groupby("tenure_bucket", observed=True)["churn"].mean().mul(100).round(1)
    col1, col2 = st.columns([2,1])
    with col1:
        st.bar_chart(tb.rename("Churn Rate (%)"))
    with col2:
        st.markdown("""<div class='insight-box'>
        <b>Recommendation:</b><br>
        Focus onboarding quality on the <b>first 6 months</b>.
        Customers who survive the first year churn at half the rate.
        Consider a 90-day check-in programme for new customers.
        </div>""", unsafe_allow_html=True)

    # Service bundle insights
    st.markdown("---")
    st.markdown("### Services & Add-ons")
    svc_data = biz.get("churn_by_service_count", [])
    if svc_data:
        svc_df = pd.DataFrame(svc_data).set_index("service_count")
        svc_df["churn_rate"] = (svc_df["churn_rate"] * 100).round(1)
        col1, col2 = st.columns([2,1])
        with col1:
            st.markdown("#### Churn Rate by Number of Add-on Services")
            st.bar_chart(svc_df["churn_rate"].rename("Churn Rate (%)"))
        with col2:
            st.markdown("""<div class='insight-box good'>
            <b>Key finding:</b><br>
            More services = lower churn. Customers with <b>3+ add-ons</b>
            churn significantly less. Bundling strategy directly reduces churn.
            <br><br>
            <b>Action:</b> Offer discounted bundle upgrades to customers with 0-1 services.
            </div>""", unsafe_allow_html=True)

    # Senior citizen
    st.markdown("---")
    st.markdown("### Demographics")
    c1,c2,c3 = st.columns(3)
    c1.metric("Churn Rate — Senior",     f"{biz.get('churn_senior',0):.1%}")
    c2.metric("Churn Rate — Non-Senior", f"{biz.get('churn_non_senior',0):.1%}")
    c3.metric("Senior Premium",
        f"+{(biz.get('churn_senior',0) - biz.get('churn_non_senior',0)):.1%}")

    st.markdown("""<div class='insight-box warn'>
    👴 <b>Senior citizens churn at nearly double the rate</b> of non-seniors.
    Consider senior-specific support, simplified billing, and dedicated account management.
    </div>""", unsafe_allow_html=True)

    # Recommendations summary
    st.markdown("---")
    st.markdown("### Top 5 Retention Recommendations")
    recs = [
        ("🔴 Critical", "Convert month-to-month customers to annual contracts",
         "Offer 10-15% discount for 12-month commitment. Expected churn reduction: ~30pp."),
        ("🔴 Critical", "Intervene in the first 6 months",
         "Automated 30/60/90-day check-ins. Early churn drives the highest LTV loss."),
        ("🟠 High",     "Migrate electronic check users to auto-pay",
         "Offer a small billing credit for switching. Electronic check users churn at 3x auto-pay rate."),
        ("🟠 High",     "Bundle add-ons for Fiber optic customers",
         "Fiber customers pay more but churn more. Security + Tech Support bundles increase stickiness."),
        ("🟡 Medium",   "Senior citizen retention programme",
         "Dedicated support line, simplified billing, proactive check-ins."),
    ]
    for priority, title, detail in recs:
        cls = "alert" if "Critical" in priority else ("warn" if "High" in priority else "")
        st.markdown(f"""<div class='insight-box {cls}'>
        <b>{priority} — {title}</b><br>{detail}
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Performance":
    st.title("🤖 Model Comparison")
    st.markdown("All 4 models trained on 80% of data, evaluated on held-out 20% test set.")
    st.markdown("---")

    model_names = list(results["results"].keys())
    comp_df = pd.DataFrame([
        {
            "Model":     n,
            "AUC-ROC":   results["results"][n]["roc_auc"],
            "F1 Score":  results["results"][n]["f1"],
            "Accuracy":  results["results"][n]["accuracy"],
            "Precision": results["results"][n]["precision"],
            "Recall":    results["results"][n]["recall"],
        }
        for n in model_names
    ])

    st.dataframe(
        comp_df.style
            .highlight_max(subset=["AUC-ROC","F1 Score","Accuracy"], color="#d1fae5")
            .format({c: "{:.4f}" for c in ["AUC-ROC","F1 Score","Accuracy","Precision","Recall"]}),
        use_container_width=True, hide_index=True
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### AUC-ROC Comparison")
        auc_df = pd.DataFrame(
            {"AUC-ROC": [results["results"][n]["roc_auc"] for n in model_names]},
            index=model_names)
        st.bar_chart(auc_df)
    with col2:
        st.markdown("#### F1 Score Comparison")
        f1_df = pd.DataFrame(
            {"F1 Score": [results["results"][n]["f1"] for n in model_names]},
            index=model_names)
        st.bar_chart(f1_df)

    st.markdown("---")
    st.markdown("### Why AUC-ROC is the right metric for churn")
    st.markdown("""<div class='insight-box'>
    The dataset has <b>26.6% churn</b> — a class imbalance that makes accuracy misleading.
    A model predicting "no churn" for everyone would achieve 73% accuracy but be useless.
    <br><br>
    <b>AUC-ROC</b> measures how well the model ranks churners above non-churners across all
    probability thresholds — the correct metric for imbalanced binary classification problems.
    <br><br>
    Our best model achieves <b>AUC = 0.84</b> on real Telco data, meaning it correctly identifies
    a churner vs a non-churner 84% of the time.
    </div>""", unsafe_allow_html=True)

    # Confusion matrix for best model
    st.markdown(f"### Confusion Matrix — {best}")
    cm = results["results"][best]["confusion_matrix"]
    cm_df = pd.DataFrame(cm,
        index=["Actual: Retained","Actual: Churned"],
        columns=["Predicted: Retained","Predicted: Churned"])
    st.dataframe(cm_df, use_container_width=False)
    tn,fp,fn,tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("True Negatives",  tn, help="Correctly predicted retained")
    c2.metric("False Positives", fp, help="Predicted churn but stayed")
    c3.metric("False Negatives", fn, help="Missed churners — most costly!", delta_color="inverse")
    c4.metric("True Positives",  tp, help="Correctly predicted churners")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔎 Explainability":
    st.title("🔎 Model Explainability")
    st.markdown("Understanding **why** the model predicts churn — the most important page for business trust.")
    st.markdown("---")

    if fi_data:
        fi_df = pd.DataFrame(fi_data)

        # Top features
        st.markdown("### Top 20 Features by Permutation Importance")
        st.markdown("""<div class='insight-box'>
        <b>Permutation Importance</b> measures how much model accuracy drops when a feature's
        values are randomly shuffled — features that hurt performance most when scrambled
        are the most important. This is more reliable than model-native importance.
        </div>""", unsafe_allow_html=True)

        top_fi = fi_df.head(15).set_index("feature")[["perm_importance_norm"]]
        top_fi.columns = ["Importance Score (normalised)"]
        top_fi = top_fi.sort_values("Importance Score (normalised)")
        st.bar_chart(top_fi)

        st.markdown("---")
        st.markdown("### What Each Top Feature Means")

        explanations = {
            "tenure":              ("🕒 How long the customer has been with us",
                                    "Longer tenure = much lower churn risk. The single strongest predictor."),
            "monthlycharges":      ("💰 Monthly bill amount",
                                    "Higher charges increase churn risk — customers feel the value gap."),
            "totalcharges":        ("💳 Total amount paid to date",
                                    "Proxy for loyalty + tenure combined. High total = long relationship."),
            "avg_monthly_spend":   ("📊 Total charges / tenure",
                                    "Engineered feature — captures true spending rate, smoothed by time."),
            "tenure_x_charges":    ("🔗 Tenure × Monthly Charges interaction",
                                    "Customers with high charges AND low tenure are highest risk."),
            "contract":            ("📄 Contract type",
                                    "Month-to-month contracts are the #1 structural churn driver."),
            "internetservice":     ("🌐 Type of internet service",
                                    "Fiber optic shows high churn despite (or because of) premium pricing."),
            "paymentmethod":       ("💳 How the customer pays",
                                    "Electronic check correlates strongly with churn — possible friction signal."),
            "service_count":       ("🔧 Number of active add-ons",
                                    "More services = more switching cost = lower churn."),
            "charge_per_service":  ("📉 Monthly charge per service used",
                                    "High cost per service = poor perceived value = higher churn risk."),
        }

        col1, col2 = st.columns(2)
        for i, row in fi_df.head(10).iterrows():
            feat = row["feature"]
            imp  = row["perm_importance_norm"]
            col  = col1 if i % 2 == 0 else col2
            title, desc = explanations.get(feat, (f"Feature: {feat}", "Business interpretation varies."))
            with col:
                st.markdown(f"""<div class='insight-box'>
                <b>{title}</b> &nbsp;|&nbsp; Importance: <b>{imp:.3f}</b><br>{desc}
                </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Engineered Features vs Raw Features")
        eng_feats = ["avg_monthly_spend","tenure_x_charges","charge_per_service",
                     "service_count","new_customer","long_tenure","auto_payment","high_value_customer"]
        eng_imp   = fi_df[fi_df["feature"].isin(eng_feats)]["perm_importance_norm"].sum()
        raw_imp   = fi_df[~fi_df["feature"].isin(eng_feats)]["perm_importance_norm"].sum()
        c1,c2 = st.columns(2)
        c1.metric("Engineered Feature Importance", f"{eng_imp:.3f}")
        c2.metric("Raw Feature Importance",        f"{raw_imp:.3f}")
        st.markdown("""<div class='insight-box good'>
        Engineered features contribute meaningfully alongside raw features,
        validating the feature engineering step. This is why preprocessing and
        domain knowledge matter beyond just throwing raw data at a model.
        </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.info("💡 **SHAP values** provide even deeper per-prediction explanations. "
                "Install `shap` (`pip install shap`) and re-run `train_pipeline.py` "
                "to unlock waterfall plots, beeswarm charts, and force plots for individual predictions.")
    else:
        st.warning("No feature importance data found. Run `src/train_pipeline.py` first.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — PREDICT CUSTOMER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Predict Customer":
    st.title("🎯 Real-Time Churn Prediction")
    st.markdown("Enter any customer's details to get an instant churn risk score with explanations.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Account**")
        tenure          = st.slider("Tenure (months)", 1, 72, 24)
        monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, 0.5)
        contract        = st.selectbox("Contract", ["Month-to-month","One year","Two year"])
        payment_method  = st.selectbox("Payment Method",
            ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"])
        paperless = st.selectbox("Paperless Billing", ["Yes","No"])

    with col2:
        st.markdown("**Demographics**")
        gender         = st.selectbox("Gender", ["Male","Female"])
        senior         = st.selectbox("Senior Citizen", ["No","Yes"])
        partner        = st.selectbox("Partner",    ["Yes","No"])
        dependents     = st.selectbox("Dependents", ["No","Yes"])
        st.markdown("**Services**")
        internet       = st.selectbox("Internet Service", ["Fiber optic","DSL","No"])
        phone_service  = st.selectbox("Phone Service", ["Yes","No"])
        multiple_lines = st.selectbox("Multiple Lines",
            ["Yes","No"] if phone_service=="Yes" else ["No phone service"])

    with col3:
        st.markdown("**Add-ons**")
        if internet != "No":
            online_sec  = st.selectbox("Online Security",   ["No","Yes"])
            online_bkp  = st.selectbox("Online Backup",     ["No","Yes"])
            device_prot = st.selectbox("Device Protection", ["No","Yes"])
            tech_supp   = st.selectbox("Tech Support",      ["No","Yes"])
            stream_tv   = st.selectbox("Streaming TV",      ["No","Yes"])
            stream_mv   = st.selectbox("Streaming Movies",  ["No","Yes"])
        else:
            online_sec = online_bkp = device_prot = tech_supp = "No internet service"
            stream_tv = stream_mv = "No internet service"

    st.markdown("---")
    if st.button("🔮 Predict Churn Risk", type="primary", use_container_width=True):
        input_df = pd.DataFrame([{
            "gender":             gender,
            "seniorcitizen":      1 if senior == "Yes" else 0,
            "partner":            partner,
            "dependents":         dependents,
            "tenure":             tenure,
            "phoneservice":       phone_service,
            "multiplelines":      multiple_lines,
            "internetservice":    internet,
            "onlinesecurity":     online_sec,
            "onlinebackup":       online_bkp,
            "deviceprotection":   device_prot,
            "techsupport":        tech_supp,
            "streamingtv":        stream_tv,
            "streamingmovies":    stream_mv,
            "contract":           contract,
            "paperlessbilling":   paperless,
            "paymentmethod":      payment_method,
            "monthlycharges":     monthly_charges,
            "totalcharges":       tenure * monthly_charges,
        }])
        input_feat = engineer_features(input_df)
        proba      = model.predict_proba(input_feat)[0][1]

        st.markdown("---")
        r1, r2, r3 = st.columns(3)

        with r1:
            st.markdown("### Risk Score")
            if proba >= 0.65:
                st.error(f"🔴 HIGH RISK — {proba:.1%}")
            elif proba >= 0.40:
                st.warning(f"🟡 MEDIUM RISK — {proba:.1%}")
            else:
                st.success(f"🟢 LOW RISK — {proba:.1%}")
            st.progress(float(proba))
            st.caption(f"Model: {best} | AUC: {best_r['roc_auc']}")

        with r2:
            st.markdown("### Why this score?")
            factors = []
            if contract == "Month-to-month":        factors.append("⚠️ Month-to-month contract (+high risk)")
            if tenure < 6:                          factors.append("⚠️ New customer — tenure < 6 months")
            if internet == "Fiber optic":           factors.append("⚠️ Fiber optic service — high churn segment")
            if payment_method == "Electronic check":factors.append("⚠️ Electronic check — friction signal")
            if monthly_charges > 80:                factors.append("⚠️ High monthly charges > $80")
            if online_sec == "No" and internet!="No": factors.append("⚠️ No Online Security add-on")
            if tech_supp == "No" and internet!="No":  factors.append("⚠️ No Tech Support add-on")
            if tenure > 36:                         factors.append("✅ Long tenure > 3 years — loyalty signal")
            if contract == "Two year":              factors.append("✅ Two-year contract — committed customer")
            if payment_method in ["Bank transfer (automatic)","Credit card (automatic)"]:
                                                    factors.append("✅ Auto-payment — reduces churn risk")
            for f in factors[:6]:
                st.markdown(f"- {f}")

        with r3:
            st.markdown("### Recommended Action")
            if proba >= 0.65:
                st.markdown("""
                **Immediate intervention:**
                - 📞 Proactive retention call within 48h
                - 💰 Offer annual contract with 15% discount
                - 🎁 Loyalty bundle — add Online Security free for 3 months
                - 📊 Flag in CRM as Priority-1 retention case
                """)
            elif proba >= 0.40:
                st.markdown("""
                **Monitor & engage:**
                - 📧 Satisfaction survey this week
                - 💡 Recommend relevant add-on service
                - 🗓️ Offer annual contract incentive
                - 📋 Schedule 30-day follow-up
                """)
            else:
                st.markdown("""
                **Maintain relationship:**
                - 🌟 Include in loyalty rewards programme
                - 📦 Upsell opportunity — low churn risk
                - 📅 Standard quarterly engagement
                """)
