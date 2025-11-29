import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import plot_tree

# ----------------------------------------------------------------------
# STREAMLIT CONFIG
# ----------------------------------------------------------------------
st.set_page_config(page_title="Heart Disease Random Forest + SHAP", layout="wide")
st.title("❤️ Heart Disease ML Explorer with Fake Data, Metrics & SHAP")

# ----------------------------------------------------------------------
# 1) GENERATE FAKE HEART DISEASE DATA
# ----------------------------------------------------------------------
st.subheader("📌 Generated Fake Heart Disease Dataset (100 rows)")

np.random.seed(42)

df = pd.DataFrame({
    "age": np.random.randint(29, 77, 100),
    "sex": np.random.randint(0, 2, 100),
    "cholesterol": np.random.randint(150, 320, 100),
    "resting bp": np.random.randint(90, 180, 100),
    "max heart rate": np.random.randint(90, 200, 100),
    "oldpeak": np.round(np.random.uniform(0, 6, 100), 2),
    "exercise angina": np.random.randint(0, 2, 100),
})

# Create synthetic target (slightly realistic pattern)
df["heart disease"] = (
    (df["age"] > 55).astype(int)
    + (df["cholesterol"] > 240).astype(int)
    + (df["max heart rate"] < 140).astype(int)
    + np.random.randint(0, 2, 100)
)
df["heart disease"] = (df["heart disease"] > 2).astype(int)

st.dataframe(df.head())

# ----------------------------------------------------------------------
# Countplot
# ----------------------------------------------------------------------
st.subheader("Distribution of Heart Disease")
fig, ax = plt.subplots()
sns.countplot(data=df, x='heart disease', ax=ax)
ax.set_title("Heart Disease Distribution")
st.pyplot(fig)

# ----------------------------------------------------------------------
# Train-test split
# ----------------------------------------------------------------------
X = df.drop("heart disease", axis=1)
y = df["heart disease"]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

# ----------------------------------------------------------------------
# Base Random Forest
# ----------------------------------------------------------------------
st.subheader("Training Base Random Forest Model")

base_rf = RandomForestClassifier(
    random_state=42,
    n_jobs=-1,
    max_depth=5,
    n_estimators=100,
    oob_score=True
)
base_rf.fit(X_train, y_train)

st.write("**OOB Score:**", base_rf.oob_score_)

# ----------------------------------------------------------------------
# GRID SEARCH
# ----------------------------------------------------------------------
with st.spinner("Running GridSearchCV..."):
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)

    params = {
        'max_depth': [2, 3, 5, 10, 20],
        'min_samples_leaf': [5, 10, 20, 50, 100],
        'n_estimators': [10, 30, 50, 100, 200]
    }

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=params,
        cv=4,
        n_jobs=-1,
        scoring="accuracy",
        verbose=0
    )

    grid_search.fit(X_train, y_train)

rf_best = grid_search.best_estimator_

st.subheader("Best Model Parameters")
st.json(grid_search.best_params_)

# ----------------------------------------------------------------------
# 2) METRICS AND CONFUSION MATRIX
# ----------------------------------------------------------------------
st.subheader("📊 Model Evaluation Metrics")

preds = rf_best.predict(X_test)

accuracy = accuracy_score(y_test, preds)
st.write(f"### ✅ Accuracy: **{accuracy:.3f}**")

st.write("### Classification Report")
st.text(classification_report(y_test, preds))

# Confusion Matrix Heatmap
cm = confusion_matrix(y_test, preds)
fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("Actual")
ax_cm.set_title("Confusion Matrix")
st.pyplot(fig_cm)

# ----------------------------------------------------------------------
# Feature Importances
# ----------------------------------------------------------------------
st.subheader("Feature Importances")
imp_df = pd.DataFrame({
    "Variable": X.columns,
    "Importance": rf_best.feature_importances_
}).sort_values(by="Importance", ascending=False)
st.dataframe(imp_df)

# ----------------------------------------------------------------------
# Tree Visualization
# ----------------------------------------------------------------------
st.subheader("🌳 Visualize Individual Trees")

tree_index = st.slider(
    "Select Tree Index",
    min_value=0,
    max_value=len(rf_best.estimators_) - 1,
    value=0
)

fig_tree, ax_tree = plt.subplots(figsize=(30, 15))
plot_tree(
    rf_best.estimators_[tree_index],
    feature_names=X.columns,
    class_names=["Disease", "No Disease"],
    filled=True,
    fontsize=8
)
st.pyplot(fig_tree)

# ----------------------------------------------------------------------
# 3) SHAP EXPLANATIONS
# ----------------------------------------------------------------------
st.subheader("🔍 SHAP Model Explainability")

explainer = shap.TreeExplainer(rf_best)
shap_values = explainer.shap_values(X_test)

# Summary Plot
st.write("### SHAP Summary Plot")
fig_shap_summary, ax_summary = plt.subplots(figsize=(10, 6))
shap.summary_plot(shap_values[1], X_test, show=False)
st.pyplot(fig_shap_summary)

# Bar Plot
st.write("### SHAP Feature Importance Bar Plot")
fig_shap_bar, ax_bar = plt.subplots(figsize=(10, 6))
shap.summary_plot(shap_values[1], X_test, plot_type="bar", show=False)
st.pyplot(fig_shap_bar)

# Force Plot for a Specific Row
st.write("### SHAP Force Plot for a Single Prediction")

row_to_explain = st.slider("Select Row from Test Set", 0, len(X_test) - 1, 0)

shap_html = shap.force_plot(
    explainer.expected_value[1],
    shap_values[1][row_to_explain, :],
    X_test.iloc[row_to_explain, :],
    matplotlib=False
)

st.components.v1.html(shap_html.html(), height=300)
