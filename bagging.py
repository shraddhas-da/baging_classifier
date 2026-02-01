import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, r2_score

# --- APP CONFIGURATION ---
st.set_page_config(page_title="Bagging Diagnostic Lab", layout="wide")

# --- GLOBAL MODE SELECTOR ---
st.sidebar.title("🚀 Lab Navigation")
app_mode = st.sidebar.radio("Select Analysis Type", ["Classification", "Regression"])

# ---------------------------------------------------------
# SECTION 1: CLASSIFICATION (YOUR ORIGINAL CODE)
# ---------------------------------------------------------
if app_mode == "Classification":
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
        if name == "Concentric Circles": return datasets.make_circles(n_samples=500, noise=noise, factor=0.5,
                                                                      random_state=42)
        return datasets.make_blobs(n_samples=500, centers=2, cluster_std=noise * 5, random_state=42)


    X, y = load_data(ds_name, noise_level)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- EXECUTION ENGINE ---
    lone_learner = base_model
    lone_learner.fit(X_train, y_train)

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

# ---------------------------------------------------------
# SECTION 2: REGRESSION
# ---------------------------------------------------------
else:
    st.title("📉 Regression: Lone Learner vs. Bagging")
    st.sidebar.header("🛠️ Bagging Regressor")

    # 1. Base Regressor Selection
    reg_base_choice = st.sidebar.selectbox("Select Base Regressor Type", ("Decision Tree", "KNN", "SVR"))

    # 2. Number of Estimators
    n_reg_estimators = st.sidebar.number_input("Enter Number of Estimators", 1, 500, 100)

    # 3. Dynamic Base Regressor Logic
    if reg_base_choice == "Decision Tree":
        reg_model = DecisionTreeRegressor()
    elif reg_base_choice == "KNN":
        reg_model = KNeighborsRegressor(n_neighbors=5)
    else:
        reg_model = SVR(kernel="rbf")

    # 4. Hyperparameters
    with st.sidebar.expander("🎒 Bagging Regressor Hyperparameters", expanded=True):
        r_max_samples = st.slider("Max Samples (%) ", 0.1, 1.0, 1.0)
        r_max_features = st.slider("Max Features (%) ", 0.1, 1.0, 1.0)
        r_bootstrap = st.checkbox("Bootstrap Samples ", value=True)
        r_boot_feat = st.checkbox("Bootstrap Features ", value=False)

    # 5. Dataset Selection
    reg_ds = st.sidebar.selectbox("Select Regression Data", ["Sine Wave", "Linear with Noise", "Friedman"])

    # Data Engine
    X_reg = np.sort(5 * np.random.rand(200, 1), axis=0)
    if reg_ds == "Sine Wave":
        y_reg = np.sin(X_reg).ravel() + np.random.normal(0, 0.1, 200)
    elif reg_ds == "Friedman":
        X_reg, y_reg = datasets.make_friedman1(n_samples=200, n_features=1, noise=1.0)
    else:
        y_reg = 0.5 * X_reg.ravel() + np.random.normal(0, 0.2, 200)

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

    # Training
    lone_reg = reg_model.fit(X_train_r, y_train_r)
    bag_reg = BaggingRegressor(estimator=reg_model, n_estimators=n_reg_estimators, max_samples=r_max_samples,
                               max_features=r_max_features, bootstrap=r_bootstrap, bootstrap_features=r_boot_feat,
                               n_jobs=-1).fit(X_train_r, y_train_r)

    # Results UI
    rc1, rc2 = st.columns(2)
    with rc1:
        st.metric("Lone Regressor R²", f"{r2_score(y_test_r, lone_reg.predict(X_test_r)):.3f}")
        fig3, ax3 = plt.subplots()
        ax3.scatter(X_reg, y_reg, color='gray', alpha=0.3)
        ax3.plot(X_reg, lone_reg.predict(X_reg), color='red', label="Single Model")
        ax3.set_title(f"Single {reg_base_choice} Regression");
        st.pyplot(fig3)

    with rc2:
        r2_bag = r2_score(y_test_r, bag_reg.predict(X_test_r))
        st.metric("Bagging Regressor R²", f"{r2_bag:.3f}",
                  delta=f"{r2_bag - r2_score(y_test_r, lone_reg.predict(X_test_r)):.3f}")
        fig4, ax4 = plt.subplots()
        ax4.scatter(X_reg, y_reg, color='gray', alpha=0.3)
        ax4.plot(X_reg, bag_reg.predict(X_reg), color='blue', linewidth=2, label="Ensemble")
        ax4.set_title(f"Bagging ({n_reg_estimators} {reg_base_choice}s)");
        st.pyplot(fig4)
