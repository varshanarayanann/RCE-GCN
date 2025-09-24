import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler



def step4_comparison():
    st.set_page_config(page_title="Baseline Model Evaluation", layout="wide")
    st.title("Step 4: Baseline Model Evaluation")
    if 'gcn_metrics' not in st.session_state:
        st.error("❌ Please train and evaluate GCN model first.")
        return
    st.markdown("""
    The app will:
    - Run **Random Forest**, **SVM**, **MLP**, and **GMM**
    - Evaluate classification performance
    - Compare metrics against your GCN model
    """)
    # Load data
    X = st.session_state['X_reduced']
    y = np.array(st.session_state['y'])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    st.subheader("Running Models...")

    # Model definitions
    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "SVM": SVC(probability=True),
        "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300),
        "GMM": GaussianMixture(
                n_components=2,
                covariance_type='diag',
                n_init=10,
                max_iter=300,
                reg_covar=1e-5,
                random_state=42
            )
    }

    results = []

    for name, model in models.items():
        if name == "GMM":
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model.fit(X_train_scaled)
            gmm_clusters = model.predict(X_test_scaled)

            # Map GMM clusters to actual labels
            cluster_map = {}
            for cluster in np.unique(gmm_clusters):
                indices = np.where(gmm_clusters == cluster)[0]
                majority_label = int(np.mean(y_test[indices]) > 0.5)
                cluster_map[cluster] = majority_label

            y_pred = np.array([cluster_map[c] for c in gmm_clusters])
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average='weighted'),
            "Recall": recall_score(y_test, y_pred, average='weighted'),
            "F1-Score": f1_score(y_test, y_pred, average='weighted')
        })

    if "gcn_metrics" in st.session_state:
        results.append(st.session_state["gcn_metrics"])

    # Display results
    results_df = pd.DataFrame(results)
    numeric_cols = results_df.select_dtypes(include=[np.number]).columns
    st.dataframe(results_df.style.format({col: "{:.3f}" for col in numeric_cols}))

    # Plot confusion matrices
    st.subheader("Confusion Matrices")

    col1, col2 = st.columns(2)
    for i, (name, model) in enumerate(models.items()):
        if name == "GMM":
            y_pred = model.predict(X_test)
        else:
            y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm)
        # cm_df = pd.DataFrame(cm, 
        #                      index=[label_mapping[i] for i in range(len(cm))],
        #                      columns=[label_mapping[i] for i in range(len(cm))])

        with [col1, col2][i % 2]:
            st.markdown(f"**{name}**")
            fig, ax = plt.subplots()
            sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_ylabel("Actual")
            ax.set_xlabel("Predicted")
            st.pyplot(fig)