import streamlit as st

def step0_introduction():
    st.set_page_config(page_title="GEM-GRAPH App: Introduction", layout="wide")

    st.title("Introduction")

    st.markdown("""
    This interactive web application is developed to:
    - **Identify differentially expressed genes (DEGs)** using customizable **Graph Convolutional Networks (GCNs)** trained on pseudo-labels from statistical tests;
    - **Classify samples (e.g., cancer vs. non-cancer)** using selected DEGs;
    - Allow users to **adjust model parameters** and choose **graph construction strategies** (Pearson or RBF).

    **Goal**: Provide researchers with a flexible, interpretable framework for **DEG analysis** and **disease classification** using graph-based learning.

    ---

    If you're ready to begin:
    👉 Navigate to the **Upload Dataset** tab to load your gene expression data and meta-data table.
    """)

    with st.expander("1. Data Input"):
        st.markdown("""
        This app requires one input file in `.csv` format:

        - **Expression Matrix** (rows: samples, columns: genes or vice versa)

        #### 1.1 Expression Matrix (Example)

        | Sample | DDX11L1 | OR4F5 | ISG15 | AGRN | ... |
        |--------|---------|-------|-------|------|-----|
        | Sample1 | 0 | 0 | 1464 | 12344 | ... |
        | Sample2 | 1 | 0 | 1544 | 13020 | ... |
        | Sample3 | 0 | 0 | 2365 | 18867 | ... |

        """)

    with st.expander("2. DEG Detection via GCN"):
        st.markdown("""
        - Computes **log2 fold change** and **-log10 p-value** between user-defined groups.
        - Applies percentile-based thresholds to create **pseudo-labels**.
        - Constructs a **gene–gene graph** using:
            - RBF similarity (`exp(-γ||x_i - x_j||²)`)
            - Pearson correlation coefficient
        - Trains a **GCN model** on this graph to identify DEGs.

        **Configurable Parameters**:
        - Hidden channels (16-512)
        - Dropout rate (0.0 - 0.9)
        - Learning rate
        - Weight Decay
        - Epochs (10 - 500)
        - Graph Edge Threshold (0.0 - 1.0)
        """)

    with st.expander("3. Cancer Classification with Selected DEGs"):
        st.markdown("""
        Once DEGs are selected, a second-stage GCN (or GAT) model performs classification:
        
        - **Binary** (Healthy vs Cancer)
        - **Multiclass** (Cancer subtypes)

        Results include:
        - Accuracy
        - Confusion matrix
        - Precision, Recall, F1-score
        """)

    with st.expander("4. Optional Filtering & Visualization"):
        st.markdown("""
        Before running models, you may:
        
        - Filter **low-expression genes** (CPM or variance)
        - Visualize **volcano plots**, **feature stats**, and **graph connectivity**
        - Save DEG lists and model outputs for downstream analysis
        """)

    st.success("👉 Go to the **Upload Dataset** page to get started.")
