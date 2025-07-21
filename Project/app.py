import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import GridSearchCV
st.set_page_config(page_title="Product Purchase Likelihood Prediction Dashboard", layout="wide")

# --- Custom CSS for styling ---
st.markdown("""
    <style>
    body {
        background-color: #f7fbfd;
    }
    .main {
        background-color: #f7fbfd;
    }
    .stApp {
        background-color: #f7fbfd;
    }
    .kpi-card {
        background: linear-gradient(90deg, #e0f7fa 0%, #e3f2fd 100%);
        border-radius: 12px;
        padding: 1.2em 1em 1em 1em;
        margin-bottom: 1em;
        box-shadow: 0 2px 8px rgba(30,144,255,0.07);
        text-align: center;
    }
    .section-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #008080;
        margin-top: 1.5em;
        margin-bottom: 0.5em;
    }
    .kpi-label {
        font-size: 1.1rem;
        color: #1E90FF;
        font-weight: 600;
    }
    .kpi-value {
        font-size: 2.2rem;
        color: #008080;
        font-weight: 700;
    }
    .stRadio > div { flex-direction: row; }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar Filters at Top ---
st.sidebar.header("Filter Data")
age_min =  int(0)
age_max =  int(0)
age_options = []
age_min_select = 0
age_max_select = 0
pp_min = 0
pp_max = 0
pp_options = []
pp_min_select = 0
pp_max_select = 0
all_genders = []
gender_select = []
# Data will be loaded and cleaned below, then these will be set



# --- Main Title ---
st.markdown("""
<h1 style='color:#008080;font-size:2.5rem;font-weight:800;margin-bottom:0.2em;'>Product Purchase Likelihood Prediction Dashboard</h1>
<hr style='border:1px solid #e0f7fa;margin-bottom:1.5em;'>
""", unsafe_allow_html=True)

# Load and clean data (same as before)
def load_data():
    df = pd.read_csv("purchase_data.csv")
    return df

def clean_data(df):
    df_clean = df.copy()
    num_cols = ["TimeOnSite", "Age", "AdsClicked", "PreviousPurchases"]
    for col in num_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        median = df_clean[col].median()
        df_clean[col] = df_clean[col].fillna(median)
    if df_clean["Gender"].isnull().sum() > 0:
        mode_gender = df_clean["Gender"].mode()[0]
        df_clean["Gender"] = df_clean["Gender"].fillna(mode_gender)
    df_clean = df_clean.dropna(subset=["Purchase"])
    df_clean["Purchase"] = df_clean["Purchase"].astype(float).round().astype(int)
    df_clean = df_clean[df_clean["Purchase"].isin([0, 1])]
    return df_clean

df = load_data()
df_clean = clean_data(df)

# Now set sidebar filter options based on cleaned data
age_min = int(df_clean["Age"].min())
age_max = int(df_clean["Age"].max())
age_options = list(range(age_min, age_max + 1))
age_min_select = st.sidebar.selectbox("Select Minimum Age", age_options, index=0)
age_max_select = st.sidebar.selectbox("Select Maximum Age", age_options, index=len(age_options)-1)
if age_min_select > age_max_select:
    st.sidebar.error("Minimum age cannot be greater than maximum age.")
age_range = (age_min_select, age_max_select)

all_genders = df_clean["Gender"].unique().tolist()
gender_select = st.sidebar.multiselect(
    "Select Gender",
    options=all_genders,
    default=all_genders
)
pp_min = int(df_clean["PreviousPurchases"].min())
pp_max = int(df_clean["PreviousPurchases"].max())
pp_options = list(range(pp_min, pp_max + 1))
pp_min_select = st.sidebar.selectbox("Select Minimum Previous Purchases", pp_options, index=0)
pp_max_select = st.sidebar.selectbox("Select Maximum Previous Purchases", pp_options, index=len(pp_options)-1)
if pp_min_select > pp_max_select:
    st.sidebar.error("Minimum previous purchases cannot be greater than maximum.")
pp_range = (pp_min_select, pp_max_select)

filtered_df = df_clean[
    (df_clean["Age"] >= age_range[0]) & (df_clean["Age"] <= age_range[1]) &
    (df_clean["Gender"].isin(gender_select)) &
    (df_clean["PreviousPurchases"] >= pp_range[0]) & (df_clean["PreviousPurchases"] <= pp_range[1])
]

# --- KPIs & Overview ---
st.markdown("<div class='section-title'>Overview & KPIs</div>", unsafe_allow_html=True)
kpi1, kpi2, kpi3 = st.columns(3)
with kpi1:
    st.metric("Purchase %", f"{filtered_df['Purchase'].mean()*100:.1f}%" if not filtered_df.empty else "-")
with kpi2:
    st.metric("Avg. Time on Site", f"{filtered_df['TimeOnSite'].mean():.2f} min" if not filtered_df.empty else "-")
with kpi3:
    st.metric("Avg. Ads Clicked", f"{filtered_df['AdsClicked'].mean():.2f}" if not filtered_df.empty else "-")

# Optional: Data preview in expander
with st.expander("Show Cleaned & Filtered Data (optional)"):
    st.dataframe(filtered_df.head(20))
    st.write(f"Total records after filtering: {len(filtered_df)}")

# --- Correlation Heatmap ---
st.markdown("<div class='section-title'>Correlation Heatmap</div>", unsafe_allow_html=True)
corr = filtered_df[["TimeOnSite", "Age", "AdsClicked", "PreviousPurchases", "Purchase"]].corr()
fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='teal', aspect="auto", title="Correlation Heatmap")
fig_corr.update_layout(margin=dict(l=20, r=20, t=40, b=20), font=dict(family="Arial", size=14, color="#222"))
st.plotly_chart(fig_corr, use_container_width=True)

# --- Demographics Analysis ---
st.markdown("<div class='section-title'>Demographics Analysis</div>", unsafe_allow_html=True)
# Pie chart: Purchase count by Gender (for Purchase==1)
purchase_by_gender = filtered_df[filtered_df["Purchase"] == 1]["Gender"].value_counts().reset_index()
purchase_by_gender.columns = ["Gender", "Count"]
fig_gender_pie = px.pie(purchase_by_gender, names="Gender", values="Count", title="Purchase Count by Gender", color_discrete_sequence=["#008080", "#1E90FF"])
fig_gender_pie.update_traces(textinfo='percent+label', pull=[0.05, 0.05], hoverinfo='label+percent+value')
fig_gender_pie.update_layout(margin=dict(l=20, r=20, t=40, b=20), font=dict(family="Arial", size=14, color="#222"))
st.plotly_chart(fig_gender_pie, use_container_width=True)

# Purchase likelihood by Age (integer bins)
age_bins = [int(df_clean["Age"].min()), 25, 33, 41, 49, 56, int(df_clean["Age"].max())+1]
age_labels = [f"{age_bins[i]}-{age_bins[i+1]-1}" for i in range(len(age_bins)-1)]
filtered_df["AgeGroup"] = pd.cut(filtered_df["Age"], bins=age_bins, labels=age_labels, include_lowest=True, right=False)
age_purchase = filtered_df.groupby("AgeGroup")["Purchase"].mean().reset_index()
fig_age = px.bar(age_purchase, x="AgeGroup", y="Purchase", title="Purchase Likelihood by Age Group", color_discrete_sequence=["#1E90FF"])
fig_age.update_traces(hovertemplate='Age Group: %{x}<br>Purchase Likelihood: %{y:.2f}')
fig_age.update_layout(margin=dict(l=20, r=20, t=40, b=20), font=dict(family="Arial", size=14, color="#222"))
st.plotly_chart(fig_age, use_container_width=True)

# --- Behavior Analysis ---
st.markdown("<div class='section-title'>Behavior Analysis</div>", unsafe_allow_html=True)
st.markdown("<div style='height:0.5em'></div>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    fig1 = px.histogram(filtered_df, x="Age", nbins=20, title="Age Distribution", color_discrete_sequence=["#008080"])
    fig1.update_traces(hovertemplate='Age: %{x}<br>Count: %{y}')
    fig1.update_layout(margin=dict(l=10, r=10, t=40, b=10), font=dict(family="Arial", size=13, color="#222"))
    st.plotly_chart(fig1, use_container_width=True)
with col2:
    fig2 = px.histogram(filtered_df, x="TimeOnSite", nbins=20, title="Time on Site Distribution", color_discrete_sequence=["#1E90FF"])
    fig2.update_traces(hovertemplate='Time on Site: %{x}<br>Count: %{y}')
    fig2.update_layout(margin=dict(l=10, r=10, t=40, b=10), font=dict(family="Arial", size=13, color="#222"))
    st.plotly_chart(fig2, use_container_width=True)
with col3:
    fig3 = px.histogram(filtered_df, x="AdsClicked", nbins=20, title="Ads Clicked Distribution", color_discrete_sequence=["#20B2AA"])
    fig3.update_traces(hovertemplate='Ads Clicked: %{x}<br>Count: %{y}')
    fig3.update_layout(margin=dict(l=10, r=10, t=40, b=10), font=dict(family="Arial", size=13, color="#222"))
    st.plotly_chart(fig3, use_container_width=True)
ads_bins = [int(df_clean["AdsClicked"].min()), 3, 6, 9, 12, 15, int(df_clean["AdsClicked"].max())+1]
ads_labels = [f"{ads_bins[i]}-{ads_bins[i+1]-1}" for i in range(len(ads_bins)-1)]
filtered_df["AdsGroup"] = pd.cut(filtered_df["AdsClicked"], bins=ads_bins, labels=ads_labels, include_lowest=True, right=False)
ads_purchase = filtered_df.groupby("AdsGroup")["Purchase"].mean().reset_index()
fig_ads = px.bar(ads_purchase, x="AdsGroup", y="Purchase", title="Purchase Likelihood by Ads Clicked", color_discrete_sequence=["#008080"])
fig_ads.update_traces(hovertemplate='Ads Clicked: %{x}<br>Purchase Likelihood: %{y:.2f}')
fig_ads.update_layout(margin=dict(l=20, r=20, t=40, b=20), font=dict(family="Arial", size=14, color="#222"))
st.plotly_chart(fig_ads, use_container_width=True)

# --- Model Training ---
st.markdown("<div class='section-title'>Model Training & Performance</div>", unsafe_allow_html=True)
# Only run modeling if enough data after filtering and target is binary
gender_cols = []  # Ensure gender_cols is defined globally
feature_cols = [] # Ensure feature_cols is defined globally
if len(filtered_df) >= 30 and set(filtered_df["Purchase"].unique()).issubset({0, 1}):
    # --- Feature Engineering ---
    filtered_df = pd.get_dummies(filtered_df, columns=["Gender"], drop_first=True)
    filtered_df["Age_TimeOnSite"] = filtered_df["Age"] * filtered_df["TimeOnSite"]
    feature_cols = ["TimeOnSite", "Age", "AdsClicked", "PreviousPurchases", "Age_TimeOnSite"]
    gender_cols = [col for col in filtered_df.columns if col.startswith("Gender_")]
    feature_cols += gender_cols
    X = filtered_df[feature_cols]
    y = filtered_df["Purchase"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # --- Hyperparameter Tuning: Logistic Regression ---
    logreg_params = {'C': [0.01, 0.1, 1, 10, 100], 'solver': ['lbfgs', 'liblinear']}
    logreg_grid = GridSearchCV(LogisticRegression(max_iter=1000), logreg_params, cv=5, scoring='roc_auc')
    logreg_grid.fit(X_train, y_train)
    logreg = logreg_grid.best_estimator_
    y_pred_logreg = logreg.predict(X_test)
    y_proba_logreg = logreg.predict_proba(X_test)[:, 1]
    # --- Hyperparameter Tuning: Decision Tree ---
    dtree_params = {'max_depth': [3, 5, 7, 10, None], 'min_samples_split': [2, 5, 10]}
    dtree_grid = GridSearchCV(DecisionTreeClassifier(random_state=42), dtree_params, cv=5, scoring='roc_auc')
    dtree_grid.fit(X_train, y_train)
    dtree = dtree_grid.best_estimator_
    y_pred_dtree = dtree.predict(X_test)
    y_proba_dtree = dtree.predict_proba(X_test)[:, 1]
    def get_metrics(y_true, y_pred, y_proba):
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1-Score": f1_score(y_true, y_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_true, y_proba)
        }
    metrics_logreg = get_metrics(y_test, y_pred_logreg, y_proba_logreg)
    metrics_dtree = get_metrics(y_test, y_pred_dtree, y_proba_dtree)
    st.subheader(":blue[Model Performance]")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Logistic Regression**")
        for k, v in metrics_logreg.items():
            st.write(f"{k}: {v:.3f}")
    with col2:
        st.markdown("**Decision Tree**")
        for k, v in metrics_dtree.items():
            st.write(f"{k}: {v:.3f}")
    fpr_logreg, tpr_logreg, _ = roc_curve(y_test, y_proba_logreg)
    fpr_dtree, tpr_dtree, _ = roc_curve(y_test, y_proba_dtree)
    plt.figure(figsize=(6,4))
    plt.plot(fpr_logreg, tpr_logreg, label='Logistic Regression', color='teal')
    plt.plot(fpr_dtree, tpr_dtree, label='Decision Tree', color='blue')
    plt.plot([0,1],[0,1],'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    st.pyplot(plt.gcf())
    plt.close()
else:
    st.warning("Not enough data after filtering, or target is not binary. Please adjust your filters or check your data.")

# --- Model Predictions (User Input) ---
st.markdown("<div class='section-title'>Predict Purchase for a New User</div>", unsafe_allow_html=True)
model_choice = st.radio("Select Model for Prediction:", ("Logistic Regression", "Decision Tree"), horizontal=True)
with st.form("predict_form"):
    st.write("#### Enter User Details:")
    input_time = st.number_input("Time on Site (minutes)", min_value=0.0, max_value=100.0, value=10.0)
    input_age = st.number_input("Age", min_value=0, max_value=120, value=30)
    input_gender = st.selectbox("Gender", all_genders)
    input_ads = st.number_input("Ads Clicked", min_value=0, max_value=20, value=2)
    input_prev = st.number_input("Previous Purchases", min_value=0, max_value=20, value=1)
    submitted = st.form_submit_button("Predict Purchase Likelihood")

if submitted:
    input_df = pd.DataFrame({
        "TimeOnSite": [input_time],
        "Age": [input_age],
        "AdsClicked": [input_ads],
        "PreviousPurchases": [input_prev]
    })
    # Encode gender columns, add missing gender columns as 0
    for col in gender_cols:
        input_df[col] = 1 if col.split("_")[1] == input_gender else 0
    input_df["Age_TimeOnSite"] = input_df["Age"] * input_df["TimeOnSite"]
    # Ensure all feature columns are present in input_df
    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = 0
    input_features = input_df[feature_cols]  # Ensure correct order
    if model_choice == "Logistic Regression":
        model = logreg
    else:
        model = dtree
    pred_proba = model.predict_proba(input_features)[0][1]
    pred_label = model.predict(input_features)[0]
    st.success(f"Predicted Purchase Likelihood: {pred_proba*100:.1f}%") 