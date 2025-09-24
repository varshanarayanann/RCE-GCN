import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from collections import Counter
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt


def step2_feature_selection():
    st.set_page_config(page_title="GEM-GRAPH App: Feature Selection", layout="wide")
    st.title("Step 2: Select Target Column and Feature Selection Method")

    if 'uploaded_df' not in st.session_state:
        st.error("❌ Please upload a dataset first.")
        return

    df = st.session_state['uploaded_df']

    # 1. Select Target Column
    target_column = st.selectbox(
        "Select the target column:",
        options=df.columns.tolist()
    )
    if target_column:
        st.info(f"🧬 Preview of selected Target Column: '{target_column}'")
        st.write(df[target_column].head(10))

        if target_column and target_column in df.columns:
            st.info(f"📊 Distribution of feature: **{target_column}**")

            try:
                feature_data = df[target_column].dropna()
                counts = feature_data.value_counts().sort_index()
                fig, ax = plt.subplots()
                counts.plot(kind='barh', ax=ax, color='skyblue')  # Horizontal bar chart
                ax.set_title(f"Distribution of feature: {target_column}")
                ax.set_xlabel("Count")
                ax.set_ylabel("Class")
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"⚠️ Unable to plot distribution: {e}")
        elif target_column:
            st.warning("⚠️ Feature not found in the dataset.")

    # Button to confirm the selected target column
    if st.button("Confirm Target Column"):
        st.session_state['target_column_confirmed'] = target_column
        st.success(f"🎯 Target column '{target_column}' confirmed.")

    # Proceed only if target column is confirmed
    if 'target_column_confirmed' in st.session_state:
        confirmed_target = st.session_state['target_column_confirmed']

        # 2. Prepare X and y
        X = df.drop(columns=[confirmed_target])
        X = X.select_dtypes(include=[np.number])  # Keep only numeric features

        y = df[confirmed_target]  # Keep as Series

        # Remove rows where y is NaN
        valid_indices = ~y.isna()
        X = X.loc[valid_indices]
        y = y.loc[valid_indices]

        # Fill missing values in X
        X = X.fillna(X.mean())

        st.write(f"🔎 Shape before feature selection: {X.shape}")

        # 2. Select Feature Selection Method
        feature_method = st.radio(
            "Choose feature selection method:",
            ("Boruta", "PCA", "Ensemble (RF + MLP + GMM) (Only meant for datasets with <5000 columns)")
        )

        # 3. Apply Selected Feature Selection
        X_reduced = None

        if feature_method == "Boruta":
            st.info("Running Boruta Feature Selection...")

            rf = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced',
                max_depth=3
            )
            boruta = BorutaPy(
                rf,
                n_estimators='auto',
                verbose=0,
                random_state=42,
                max_iter=10
            )
            boruta.fit(X.values, y)
            selected_features = np.where(boruta.support_)[0]
            X_reduced = X.iloc[:, selected_features]

            st.success(f"✅ Boruta selected {X_reduced.shape[1]} features.")

        elif feature_method == "PCA":
            st.info("Running PCA Feature Selection...")

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            pca = PCA(n_components=0.95)  # Keep 95% variance
            X_pca = pca.fit_transform(X_scaled)
            X_reduced = pd.DataFrame(X_pca)

            st.success(f"✅ PCA reduced features to {X_reduced.shape[1]} components.")

        elif feature_method == "Ensemble (RF + MLP + GMM)":
            st.info("Running Ensemble Feature Selection...")

            features = []
            for gene in X.columns:
                expr_class1 = X[y == 1][gene]
                expr_class0 = X[y == 0][gene]
                mean_class1 = expr_class1.mean()
                mean_class0 = expr_class0.mean()
                log2fc = np.log2((mean_class1 + 1e-8) / (mean_class0 + 1e-8))
                pval = ttest_ind(expr_class1, expr_class0, equal_var=False).pvalue
                features.append([log2fc, pval])

            X_gene = pd.DataFrame(features, columns=['log2fc', 'pval'])
            X_gene['log_pval'] = -np.log10(np.clip(X_gene['pval'], 1e-10, 1.0))
            X_gene = X_gene.drop(columns=['pval'])

            y_gene = np.array([1 if abs(lfc) > 1 else 0 for lfc in X_gene['log2fc']])

            threshold = 0.32

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_gene)

            # Random Forest
            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=3,
                random_state=42,
                class_weight='balanced'
            )
            rf.fit(X_scaled, y_gene)
            rf_pred = (rf.predict_proba(X_scaled)[:, 1] > threshold).astype(int)

            # MLP
            mlp_pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('mlp', MLPClassifier(hidden_layer_sizes=(32, 16), activation='relu', max_iter=500, random_state=42))
            ])
            mlp_pipeline.fit(X_gene, y_gene)
            mlp_pred = (mlp_pipeline.predict_proba(X_gene)[:, 1] > threshold).astype(int)

            # GMM
            gmm = GaussianMixture(
                n_components=2,
                covariance_type='diag',
                n_init=10,
                max_iter=300,
                reg_covar=1e-5,
                random_state=42
            )
            gmm.fit(X_scaled)
            gmm_clusters = gmm.predict(X_scaled)
            gmm_map = {}
            for cluster in np.unique(gmm_clusters):
                genes_in_cluster = np.where(gmm_clusters == cluster)[0]
                majority_label = int(np.mean(y_gene[genes_in_cluster]) > 0.5)
                gmm_map[cluster] = majority_label
            gmm_pred = np.array([gmm_map[c] for c in gmm_clusters])

            # Hard Voting
            ensemble_preds = []
            for r, m, g in zip(rf_pred, mlp_pred, gmm_pred):
                votes = [r, m, g]
                majority_vote = Counter(votes).most_common(1)[0][0]
                ensemble_preds.append(majority_vote)

            ensemble_preds = np.array(ensemble_preds)

            # Select genes where ensemble_preds == 1
            selected_gene_indices = np.where(ensemble_preds == 1)[0]
            selected_gene_names = X.columns[selected_gene_indices]

            X_reduced = X[selected_gene_names]

            st.success(f"✅ Ensemble selected {X_reduced.shape[1]} genes.")

        # Save reduced data to session for next steps
        st.session_state['X_reduced'] = X_reduced
        st.session_state['y'] = y

        st.write(f"🎯 Final shape after feature selection: {X_reduced.shape}")
        st.success(f"Please Proceed to the Model Training and Results step")

    else:
        st.info("Please select and confirm your target column to continue.")
