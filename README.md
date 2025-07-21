# Product Purchase Likelihood Prediction Dashboard

This project is an interactive Streamlit dashboard that predicts the likelihood of a user making a product purchase based on demographic and behavioral data. It uses machine learning models (Logistic Regression and Decision Tree) to provide predictions and insights, and offers a range of data visualizations and analytics.

## Features
- **Data Filtering:** Filter dataset by age, gender, and previous purchases using the sidebar.
- **KPIs & Overview:** View key metrics such as purchase percentage, average time on site, and ads clicked.
- **Visual Analytics:**
  - Correlation heatmap
  - Demographics analysis (pie and bar charts)
  - Behavioral analysis (histograms, purchase likelihood by ads clicked)
- **Model Training & Performance:**
  - Trains and evaluates Logistic Regression and Decision Tree models
  - Hyperparameter tuning with GridSearchCV
  - Displays accuracy, precision, recall, F1-score, ROC-AUC, and ROC curves
- **Predict for New User:**
  - Enter new user details and predict purchase likelihood using the selected model

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <repo-url>
cd Project
```

### 2. Install Dependencies
It is recommended to use a virtual environment.
```bash
pip install -r requirements.txt
```

### 3. Prepare the Data
Ensure `purchase_data.csv` is present in the project directory. This file should contain the following columns:
- `TimeOnSite`
- `Age`
- `Gender`
- `AdsClicked`
- `PreviousPurchases`
- `Purchase` (target: 0 or 1)

### 4. Run the Dashboard
```bash
streamlit run app.py
```

The dashboard will open in your browser. Use the sidebar to filter data and explore the analytics and prediction features.

## Requirements
- Python 3.7+
- See `requirements.txt` for all Python dependencies (includes: streamlit, pandas, numpy, scikit-learn, matplotlib, seaborn, plotly)

## Usage
- Adjust filters in the sidebar to explore different segments of the data.
- Review KPIs and visualizations for insights.
- Scroll down to see model performance metrics and ROC curves.
- Use the prediction form at the bottom to estimate purchase likelihood for a new user.

## Customization
- You can replace `purchase_data.csv` with your own dataset, as long as it contains the required columns.
- Modify or extend the models and visualizations in `app.py` as needed.

## License
This project is for educational and demonstration purposes. 
