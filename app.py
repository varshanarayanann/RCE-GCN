# --- IMPORTS ---
import streamlit as st
import numpy as np
import pandas as pd
import torch
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
from sklearn.metrics import classification_report
import warnings
import base64
from pathlib import Path
# Import core functions from the new pipeline module
from pipeline import (
    set_seed,
    build_sample_graph,
    GCNModel,
    recursive_gene_elimination,
    validate_gene_set
)

# Ignore specific warnings that are not critical for the app's functionality
warnings.filterwarnings("ignore", category=UserWarning)


# --- UTILITY FUNCTIONS FOR STREAMLIT ---

def plot_gene_correlation_graph(expression_df, selected_genes, corr_threshold, plot_threshold=None):
    """
    Generates and displays a gene-gene correlation graph.
    - Nodes are the selected genes.
    - Edges are based on Pearson Correlation Coefficient (PCC).
    """
    st.subheader("Gene-Gene Correlation Network")
    st.info("Nodes are the selected genes. Edges show strong correlations between them.")

    # Use a different threshold for plotting to make the graph cleaner
    if plot_threshold is None:
        plot_threshold = corr_threshold

    filtered_df = expression_df[selected_genes]
    corr_matrix = filtered_df.corr(method='pearson')

    # Build the graph
    G = nx.Graph()
    for gene in selected_genes:
        G.add_node(gene)

    edges_to_add = []
    for i in range(len(selected_genes)):
        for j in range(i + 1, len(selected_genes)):
            gene1 = selected_genes[i]
            gene2 = selected_genes[j]
            correlation = corr_matrix.loc[gene1, gene2]
            if abs(correlation) > plot_threshold:
                edges_to_add.append((gene1, gene2, {'weight': correlation}))

    G.add_edges_from(edges_to_add)

    if len(G.edges) == 0:
        st.warning("No correlations found above the plotting threshold. Try lowering the threshold or check your data.")
        return

    # Create figure and axis for plotting
    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.circular_layout(G)

    # Draw nodes and edges
    nx.draw_networkx_nodes(G, pos, node_size=200, node_color='skyblue', ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.6, ax=ax)

    ax.set_title("Gene Correlation Graph (PCC > {})".format(plot_threshold))
    st.pyplot(fig)


def plot_integrated_gradients_scores(genes, scores):
    """
    Generates and displays a bar chart of Integrated Gradient scores.
    """
    st.subheader("Integrated Gradients Feature Importance")
    st.info(
        "The bar chart shows the relative importance of each selected gene for the model's classification decisions.")

    # Create a DataFrame for easy plotting
    scores_df = pd.DataFrame({'Gene': genes, 'Score': scores})
    scores_df = scores_df.sort_values(by='Score', ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='Score', y='Gene', data=scores_df, ax=ax)
    ax.set_title("Gene Importance Scores")
    ax.set_xlabel("Integrated Gradient Score")
    ax.set_ylabel("Selected Genes")
    st.pyplot(fig)


@st.cache_data
def run_main_pipeline(_df, _seeds, _validation_seeds, corr_threshold, hidden_dim, epochs, elimination_rate,
                      min_genes_to_keep, label_col, _status_placeholder):
    """
    This function wraps the entire analysis pipeline.
    It's no longer cached to allow for live UI updates.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _status_placeholder.write(f"Device: {device}")

    all_accuracies, all_gene_counts, all_gene_sets, all_gene_scores = [], [], [], []

    # 1. Run Recursive Gene Elimination for each seed
    with _status_placeholder.container():
        st.subheader("1. Recursive Gene Elimination")
        progress_bar = st.progress(0)

        for i, seed in enumerate(_seeds):
            _status_placeholder.write(f"Starting pipeline run with Random State: {seed}...")

            best_val_accuracy, gene_count, best_genes, best_genes_ig_scores = recursive_gene_elimination(_df, device,
                                                                                                         hidden_dim,
                                                                                                         epochs,
                                                                                                         elimination_rate,
                                                                                                         min_genes_to_keep,
                                                                                                         seed,
                                                                                                         label_col,
                                                                                                         _status_placeholder,
                                                                                                         progress_bar)

            progress_bar.progress(100)  # Ensure it reaches 100%
            st.success(
                f"Run {i + 1} complete. Best validation accuracy: {best_val_accuracy:.4f} with {gene_count} genes.")

            all_accuracies.append(best_val_accuracy)
            all_gene_counts.append(gene_count)
            all_gene_sets.append(best_genes)
            all_gene_scores.append(best_genes_ig_scores)

    # Find the best set of genes across all seeds
    best_accuracy, best_gene_count, best_run_genes = -1, float('inf'), []
    for i in range(len(all_accuracies)):
        if all_accuracies[i] > best_accuracy or (
                all_accuracies[i] == best_accuracy and all_gene_counts[i] < best_gene_count):
            best_accuracy = all_accuracies[i]
            best_gene_count = all_gene_counts[i]
            best_run_genes = all_gene_sets[i]
            best_genes_ig_scores = all_gene_scores[i]

    # 2. Final Validation of the best gene set
    if best_run_genes:
        with _status_placeholder.container():
            st.subheader("2. Final Validation")
            st.info("Validating optimal gene set on the held-out TEST set...")
            validation_accuracies = []
            macro_precisions, macro_recalls, macro_f1s = [], [], []
            all_y_true_aggregated, all_y_pred_aggregated = [], []

            for seed in _validation_seeds:
                acc, y_true, y_pred = validate_gene_set(_df, best_run_genes, device, hidden_dim, epochs, seed=seed,
                                                        label_col=label_col)
                validation_accuracies.append(acc)
                all_y_true_aggregated.extend(y_true)
                all_y_pred_aggregated.extend(y_pred)

                report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
                macro_precisions.append(report_dict['macro avg']['precision'])
                macro_recalls.append(report_dict['macro avg']['recall'])
                macro_f1s.append(report_dict['macro avg']['f1-score'])

        # Generate a final consolidated classification report
        final_report = classification_report(all_y_true_aggregated, all_y_pred_aggregated,
                                             target_names=[str(c) for c in np.unique(all_y_true_aggregated)],
                                             zero_division=0, output_dict=True)
        final_report_str = classification_report(all_y_true_aggregated, all_y_pred_aggregated,
                                                 target_names=[str(c) for c in np.unique(all_y_true_aggregated)],
                                                 zero_division=0)

        return {
            "best_genes": best_run_genes,
            "best_val_accuracy": best_accuracy,
            "final_test_accuracy": np.mean(validation_accuracies),
            "final_test_accuracy_std": np.std(validation_accuracies),
            "final_macro_precision": np.mean(macro_precisions),
            "final_macro_precision_std": np.std(macro_precisions),
            "final_macro_recall": np.mean(macro_recalls),
            "final_macro_recall_std": np.std(macro_recalls),
            "final_macro_f1": np.mean(macro_f1s),
            "final_macro_f1_std": np.std(macro_f1s),
            "final_report": final_report,
            "final_report_str": final_report_str,
            "y_true": all_y_true_aggregated,
            "y_pred": all_y_pred_aggregated,
            "best_genes_ig_scores": best_genes_ig_scores
        }
    return None


def load_base64_image(image_path):
    img_bytes = Path(image_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

logo_base64 = load_base64_image("assets/xu-logo2.png")


st.set_page_config(
    page_title="RGE-GCN",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Reduce top padding (optional but makes layout tight)
st.markdown("""
    <style>
        .block-container {
            padding-top: 2.8rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Display centered logo
st.markdown(
    f"""
    <div style="text-align: center; margin-bottom: -10px;">
        <img src="data:image/png;base64,{logo_base64}" width="140">
    </div>
    """,
    unsafe_allow_html=True
)

# Center the title
st.markdown(
    "<h1 style='text-align: center;'>RGE-GCN: Recursive Gene Elimination with Graph Convolutional Networks</h1>",
    unsafe_allow_html=True
)


st.markdown("""
---

### 🧬 What is RCE-GCN?

**RCE-GCN (Recursive Gene Elimination with Graph Convolutional Networks)** is a genomic analysis framework that identifies the most informative genes for disease classification.

This application allows you to:

- Upload a patient-level gene expression dataset  
- Automatically identify the **target column** (label) or manually select it  
- Explore the dataset (first 10 rows or full dataset view)  
- Visualize distributions for any gene column  
- Run the full **RCE-GCN pipeline**, which includes:  
  - Graph construction using Pearson correlation  
  - GCN training with customizable hyperparameters  
  - Integrated Gradients for gene importance  
  - Recursive elimination of low-importance genes  
  - Multi-seed evaluation for robustness  
- View:
  - Validation/Test accuracy  
  - Final number of selected genes  
  - Classification reports  
  - Downloadable gene list  
  - IG-based top-gene bar plot  
  - Gene-gene correlation network  

---

### 🔍 Additional Methods (Experimental)

Under **“Explore Additional Method”**, you can also try:

- Boruta feature selection  
- PCA dimensionality reduction  
- Training a GCN using different embedding strategies  
- Comparing performance with:  
  - Random Forest  
  - SVM  
  - MLP  
  - Gaussian Mixture Models  

These methods were part of the research experiments for the RCE-GCN paper.

---

### 📂 Data Requirements

Your input CSV must have:

- **Rows = patients/samples**
- **Columns = genes**
- **One target column** (e.g., `label`, `target`, `y`)

Example:

| PatientID | Gene_A | Gene_B | Gene_C | ... | label |
|-----------|--------|--------|--------|-----|--------|
| Patient_1 | 0.98   | 1.45   | 0.23   | ... | 1 |
| Patient_2 | 1.12   | 0.88   | 0.56   | ... | 0 |
| Patient_3 | 0.45   | 1.99   | 1.04   | ... | 0 |

---

""")

# Sidebar for controls and parameters
with st.sidebar:
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    st.markdown("---")
    st.subheader("Developed By")

    col_c, col_d = st.columns([1, 5])
    with col_c:
        st.image("https://cdn-icons-png.flaticon.com/512/25/25231.png", width=22)
    with col_d:
        st.markdown("[**Varsha Narayanan**](https://github.com/varshanarayanann)  [📧](mailto:vn299@njit.edu)")

    col_a, col_b = st.columns([1, 5])
    with col_a:
        st.image("https://cdn-icons-png.flaticon.com/512/25/25231.png", width=22)  # GitHub icon
    with col_b:
        st.markdown("[**Shreyas Shende**](https://github.com/ShreyasShende3)  [📧](mailto:ss4897@njit.edu)")
    st.markdown("---")

    if uploaded_file:
        st.header("2. Pipeline Parameters")

        # Sliders for hyperparameters
        hidden_dim = st.slider("Hidden Layer Dimensions", 16, 128, 64)
        epochs = st.slider("Training Epochs", 50, 500, 200)
        elimination_rate = st.slider("Elimination Rate (%)", 0.01, 0.5, 0.1, step=0.01)
        corr_threshold = st.slider("Graph Edge Threshold (Correlation)", 0.0, 1.0, 0.7, step=0.05)
        plot_corr_threshold = st.slider("Gene Graph Plot Threshold (PCC)", 0.0, 1.0, 0.5, step=0.05)

        # Multi-select for random seeds
        default_seeds = [42, 100, 200, 300, 400]
        random_seeds = st.multiselect("Random Seeds for Gene Selection", options=default_seeds, default=default_seeds)
        validation_seeds = st.multiselect("Random Seeds for Final Validation", options=[1, 2, 3, 4, 5],
                                          default=[1, 2, 3, 4, 5])

        run_button = st.button("Start Analysis")

# Main content area
if uploaded_file:
    try:
        # Load the dataframe
        df = pd.read_csv(uploaded_file, index_col=0)

        # User selects the label column
        label_col = st.selectbox("Select the target/label column", options=df.columns,
                                 index=df.columns.get_loc('label') if 'label' in df.columns else df.columns.get_loc(
                                     'Y') if 'Y' in df.columns else 0)

        # Display data preview and visualization options
        st.header("Dataset Overview")
        preview_tab, visualize_tab = st.tabs(["Data Preview", "Data Visualization"])

        with preview_tab:
            st.write("First 10 rows of the dataset:")
            st.dataframe(df.head(10))
            show_full_df = st.checkbox("Show entire dataset")
            if show_full_df:
                st.dataframe(df)

        with visualize_tab:
            vis_cols = st.multiselect("Select columns to visualize their distribution", options=df.columns)
            if vis_cols:
                for col in vis_cols:
                    st.subheader(f"Distribution of '{col}'")
                    fig, ax = plt.subplots()
                    if pd.api.types.is_numeric_dtype(df[col]):
                        sns.histplot(df[col], kde=True, ax=ax)
                    else:
                        sns.countplot(x=df[col], ax=ax)
                    st.pyplot(fig)

        if run_button:
            # Check if a label column is selected and if it's the only one left
            if len(df.drop(columns=[label_col]).columns) == 0:
                st.error("There are no feature columns left after selecting the label. Please check your dataset.")
            else:
                # Use a status container to show live updates
                with st.status("Running Analysis...", expanded=True) as status:
                    st.write("Starting the gene selection pipeline...")

                    # Calculate min genes based on original gene count
                    original_gene_count = df.drop(columns=label_col).shape[1]
                    min_genes = int(original_gene_count * 0.05)
                    if min_genes == 0:
                        min_genes = 1

                    # Run the full pipeline and get results
                    results = run_main_pipeline(df, random_seeds, validation_seeds, corr_threshold, hidden_dim, epochs,
                                                elimination_rate, min_genes, label_col, status)

                    status.update(label="Analysis Complete!", state="complete", expanded=False)

                if results:
                    # Display Results
                    st.header("Analysis Results")

                    # Use columns for a clean, side-by-side layout
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Best Validation Accuracy", f"{results['best_val_accuracy']:.4f}")
                    with col2:
                        st.metric("Avg. Test Accuracy",
                                  f"{results['final_test_accuracy']:.4f} ± {results['final_test_accuracy_std']:.4f}")
                    with col3:
                        st.metric("Final Gene Count", len(results['best_genes']))

                    # Expander for detailed classification report and metrics
                    with st.expander("Detailed Metrics & Reports"):
                        st.subheader("Aggregated Classification Report (over all validation runs)")
                        st.code(results['final_report_str'])

                        st.subheader("Key Performance Metrics")
                        st.write(
                            f"- Average Macro Precision: {results['final_macro_precision']:.4f} ± {results['final_macro_precision_std']:.4f}")
                        st.write(
                            f"- Average Macro Recall: {results['final_macro_recall']:.4f} ± {results['final_macro_recall_std']:.4f}")
                        st.write(
                            f"- Average Macro F1-Score: {results['final_macro_f1']:.4f} ± {results['final_macro_f1_std']:.4f}")

                    # Display the list of top genes
                    with st.expander("Top Selected Genes"):
                        st.write("Here are the genes selected by the pipeline:")
                        st.dataframe(pd.DataFrame({"Gene": results['best_genes']}))

                    # New Visualizations
                    st.header("New Visualizations")

                    # Plot the gene-gene correlation graph
                    plot_gene_correlation_graph(df, results['best_genes'], corr_threshold,
                                                plot_threshold=plot_corr_threshold)

                    # Plot the integrated gradients scores
                    plot_integrated_gradients_scores(results['best_genes'], results['best_genes_ig_scores'])

    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    st.info("Please upload a CSV file to begin the analysis.")
st.markdown(
    """
    <hr style="margin-top: 10px; margin-bottom: 10px;">
    <div style="text-align: center; color: gray; font-size: 14px; padding: 0; margin: 0;">
        © Xu Lab 2025
    </div>
    """,
    unsafe_allow_html=True
)