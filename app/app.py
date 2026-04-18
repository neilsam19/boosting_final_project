import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from adaboost import AdaBoost
from modifications import AdaBoostClipped, AdaBoostPersistent, AdaBoostSoft
from data import prepare_data

st.title("Exploring AdaBoost Variants Under Noise")
st.markdown(
    "Compare how Vanilla AdaBoost and its modified variants behave "
    "as label noise increases. Select a dataset, noise level, and method in the sidebar."
)

st.sidebar.header("Controls")

dataset = st.sidebar.selectbox("Dataset", ["moons", "classification", "digits", "wine"])
noise = st.sidebar.slider("Noise level", 0.0, 0.4, 0.0, step=0.05)
method = st.sidebar.selectbox("Method", ["Vanilla", "Clipped", "Persistent", "Soft"])
T = st.sidebar.slider("Boosting rounds (T)", 5, 100, 50)

st.sidebar.subheader("Method parameters")
if method == "Clipped":
    cap = st.sidebar.slider("cap", 0.001, 0.1, 0.01, step=0.001, format="%.3f")
elif method == "Persistent":
    threshold = st.sidebar.slider("threshold", 1, 10, 2)
    damp_factor = st.sidebar.slider("damp_factor", 0.1, 1.0, 0.6, step=0.05)
elif method == "Soft":
    beta = st.sidebar.slider("beta", 0.1, 1.0, 0.5, step=0.05)


def build_model(method, T, **kwargs):
    if method == "Vanilla":
        return AdaBoost(T=T)
    elif method == "Clipped":
        return AdaBoostClipped(T=T, cap=kwargs["cap"])
    elif method == "Persistent":
        return AdaBoostPersistent(T=T, threshold=kwargs["threshold"], damp_factor=kwargs["damp_factor"])
    elif method == "Soft":
        return AdaBoostSoft(T=T, beta=kwargs["beta"])


def train_model(dataset, noise, method, T, **kwargs):
    X_train, X_test, y_train, y_test = prepare_data(dataset, noise_level=noise)
    model = build_model(method, T, **kwargs)
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test


def plot_decision_boundary(model, X_train, y_train):
    x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
    y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="RdBu")
    ax.contour(xx, yy, Z, colors="k", linewidths=0.8, levels=[0])
    pos = y_train == 1
    ax.scatter(X_train[pos, 0], X_train[pos, 1], c="steelblue", edgecolors="k", s=30, label="+1")
    ax.scatter(X_train[~pos, 0], X_train[~pos, 1], c="tomato", edgecolors="k", s=30, label="-1")
    ax.set_title("Decision Boundary")
    ax.legend(fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig


def plot_weight_scatter(X_train, y_train, weights):
    fig, ax = plt.subplots(figsize=(6, 4))
    sizes = weights / weights.max() * 300
    pos = y_train == 1
    ax.scatter(X_train[pos, 0], X_train[pos, 1], s=sizes[pos], c="steelblue", edgecolors="k", alpha=0.7, label="+1")
    ax.scatter(X_train[~pos, 0], X_train[~pos, 1], s=sizes[~pos], c="tomato", edgecolors="k", alpha=0.7, label="-1")
    ax.set_title("Weight Distribution (final round)")
    ax.legend(fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig


kwargs = {}
if method == "Clipped":
    kwargs["cap"] = cap
elif method == "Persistent":
    kwargs["threshold"] = threshold
    kwargs["damp_factor"] = damp_factor
elif method == "Soft":
    kwargs["beta"] = beta

with st.spinner("Training..."):
    model, X_train, X_test, y_train, y_test = train_model(dataset, noise, method, T, **kwargs)

preds = model.predict(X_test)
accuracy = np.mean(preds == y_test)

st.subheader(f"Test Accuracy: {accuracy:.2%}")

rounds = list(range(1, T + 1))

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(rounds, model.training_errors, color="steelblue")
    ax.set_xlabel("Boosting Round")
    ax.set_ylabel("Training Error")
    ax.set_title("Training Error Curve")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(rounds, model.max_weights, color="darkorange")
    ax.set_xlabel("Boosting Round")
    ax.set_ylabel("Max Weight")
    ax.set_title("Max Weight Over Time")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

if dataset == "moons":
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Decision Boundary")
        st.pyplot(plot_decision_boundary(model, X_train, y_train))
    with col4:
        st.subheader("Weight Visualization")
        final_weights = model.weight_history[-1]
        st.pyplot(plot_weight_scatter(X_train, y_train, final_weights))