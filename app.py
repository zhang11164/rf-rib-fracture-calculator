# -*- coding: utf-8 -*-
"""Online risk calculator for the final random forest model (5 predictors).
Research use only - investigational, pending external validation.
"""
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Traumatic rib fracture - RF risk calculator", layout="centered")
st.title("Risk calculator: final random forest model")
st.caption("Predictors: age, PaO2, PaO2/FiO2 ratio, pneumonia, ISS")

@st.cache_resource
def load():
    with open("model_rf.pkl", "rb") as fh:
        return pickle.load(fh)

d = load()
FEATS = d["features"]

age = st.number_input("Age (years)", min_value=18, max_value=100, value=60)
pao2 = st.number_input("PaO2 (mmHg)", min_value=20.0, max_value=500.0, value=80.0)
pf = st.number_input("PaO2/FiO2 ratio", min_value=20.0, max_value=600.0, value=250.0)
pneu = st.selectbox("Pneumonia", ["No", "Yes"], index=1)
iss = st.number_input("ISS", min_value=1, max_value=75, value=25)

if st.button("Calculate risk", type="primary"):
    x = pd.DataFrame({"age": [age], "PaO2": [pao2], "PF_ratio": [pf],
                      "pneumonia": [1 if pneu == "Yes" else 0], "ISS": [iss]})[FEATS]
    xi = pd.DataFrame(d["scaler"].transform(d["imputer"].transform(x)), columns=FEATS)
    p = float(d["model"].predict_proba(xi)[:, 1][0])
    st.metric("Predicted risk of the study endpoint", "%.1f%%" % (p * 100))
    st.progress(min(max(p, 0.0), 1.0))
    if p >= 0.40:
        st.warning("Above the prespecified threshold of 40%.")
    explainer = shap.TreeExplainer(d["model"])
    sv = np.asarray(explainer(xi).values)[0, :, 1]
    base = float(np.asarray(explainer(xi).base_values)[0, 1])
    order = np.argsort(-np.abs(sv))
    fig, ax = plt.subplots(figsize=(6, 3.2))
    left = base
    for k in order:
        v = sv[k]
        ax.bar(FEATS[k], v, bottom=left, color="#2166AC" if v > 0 else "#D6604D")
        left += v
    ax.axhline(base, color="gray", ls=":", lw=1)
    ax.set_ylabel("Predicted probability")
    ax.set_title("SHAP contribution (base rate %.3f -> predicted %.3f)" % (base, p))
    fig.tight_layout()
    st.pyplot(fig)

st.caption("Research use only - investigational pending external validation.")
