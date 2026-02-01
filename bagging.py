import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# --- APP CONFIGURATION ---
st.set_page_config(page_title="Bagging Diagnostic Lab", layout="wide")

st.sidebar.header("🛠️ Bagging Classifier")

# 1. Base Estimator Selection
base_choice = st.sidebar.selectbox(
    "Select Base Estimator Type",
    ("Decision Tree", "KNN", "SVM")
)

# 2. Number of Estimators
n_estimators = st.sidebar.number_input("Enter Number of Estimators", min_value=1, max_value=500, value=100)

# 3. Dynamic Base Estimator Parameters
if base_choice == "Decision Tree":
    base_model = DecisionTreeClassifier(max_depth=None)
    description = "High-variance learner; prone to overfitting."
elif base_choice == "KNN":
    base_model = KNeighborsClassifier(n_neighbors=3)
    description = "Local manifold learner; sensitive to local noise."
else:
    base_model = SVC(kernel="rbf", probability=True)
    description = "Maximum margin learner; stable but computationally intensive."

# 4. Bagging Hyperparameters
with st.sidebar.expander("🎒 Bagging Hyperparameters", expanded=True):
    max_samples = st.slider("Max Samples (%)", 0.1, 1.0, 0.8)
    max_features = st.slider("Max Features (%)", 0.1, 1.0, 1.0)
    bootstrap = st.checkbox("Bootstrap Samples (With Replacement)", value=True)
    bootstrap_features = st.checkbox("Bootstrap Features", value=False)

# 5. Dataset Selection
st.sidebar.header("📊 Data Domain")
ds_name = st.sidebar.selectbox("Select Challenge",
    ["Interlocking Moons", "Concentric Circles", "Gaussian Blobs"])
noise_level = st.sidebar.slider("Data Noise Level", 0.05, 0.5, 0.2)

# --- DATA ENGINE ---
def load_data(name, noise):
    if name == "Interlocking Moons": return datasets.make_moons(n_samples=500, noise=noise, random_state=42)
    if name == "Concentric Circles": return datasets.make_circles(n_samples=500, noise=noise, factor=0.5, random_state=42)
    return datasets.make_blobs(n_samples=500, centers=2, cluster_std=noise*5, random_state=42)

X, y = load_data(ds_name, noise_level)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- EXECUTION ENGINE ---

# Model A: The Lone Learner
lone_learner = base_model
lone_learner.fit(X_train, y_train)

# Model B: The Bagging Ensemble
ensemble = BaggingClassifier(
    estimator=base_model,
    n_estimators=n_estimators,
    max_samples=max_samples,
    max_features=max_features,
    bootstrap=bootstrap,
    bootstrap_features=bootstrap_features,
    n_jobs=-1,
    random_state=42
)
ensemble.fit(X_train, y_train)

# --- VISUALIZATION ---
st.title("🛡️ The Ensemble Effect: Individual vs. Bagging")
st.write(f"Evaluating **{base_choice}** logic: {description}")

def plot_boundary(model, X, y, ax, title):
    h = .02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='RdYlBu', s=20, alpha=0.5)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axis('off')

col1, col2 = st.columns(2)

with col1:
    acc_lone = accuracy_score(y_test, lone_learner.predict(X_test))
    st.metric("Lone Learner Accuracy", f"{acc_lone:.2%}")
    fig1, ax1 = plt.subplots()
    plot_boundary(lone_learner, X, y, ax1, f"Single {base_choice}")
    st.pyplot(fig1)

with col2:
    acc_ens = accuracy_score(y_test, ensemble.predict(X_test))
    st.metric("Bagging Ensemble Accuracy", f"{acc_ens:.2%}", delta=f"{acc_ens - acc_lone:.2%}")
    fig2, ax2 = plt.subplots()
    plot_boundary(ensemble, X, y, ax2, f"Bagging ({n_estimators} {base_choice}s)")
    st.pyplot(fig2)

st.info(f"**Insight:** Bagging with {n_estimators} estimators uses {'bootstrap' if bootstrap else 'full'} sampling to create diverse perspectives, significantly smoothing the decision boundary.")