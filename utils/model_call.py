import os
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from scipy.stats import ttest_ind
import torch.nn as nn
from torch_geometric.nn import GCNConv
from sklearn.utils.class_weight import compute_class_weight
import torch.nn.functional as F
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def generate_pseudo_labels(X, y, fc_percentile=85, pval_percentile=15):
    """
    For real data: generate pseudo-labels using thresholds on log2fc and p-value,
    and return enriched gene features.
    Args:
        X: (n_samples, n_genes) DataFrame of gene expression.
        y: binary class labels (0/1)
        fc_percentile: percentile for abs(log2fc)
        pval_percentile: percentile for -log10(pval)
    Returns:
        features: (n_genes, n_features) matrix
        pseudo_labels: binary DEG labels
        gene_names: list of gene names
        fc_threshold: actual fold change threshold used
        pval_threshold: actual p-value threshold used (in -log10 scale)
    """
    stats_arr, gene_names = compute_gene_stats(X, y, return_raw=False, add_features=False)
    # Use first two columns: [abs(log2fc), log10pval]
    log2fc_abs = stats_arr[:, 0]
    log10pval = stats_arr[:, 1]
    # Thresholds
    log2fc_thresh = np.percentile(log2fc_abs, fc_percentile)
    log10pval_thresh = np.percentile(log10pval, pval_percentile)
    pseudo_labels = ((log2fc_abs >= log2fc_thresh) & (log10pval >= log10pval_thresh)).astype(int)
    
    # Convert log10pval threshold back to p-value for plotting
    pval_thresh = 10**(-log10pval_thresh)
    
    return stats_arr, pseudo_labels, gene_names, log2fc_thresh, pval_thresh

def build_patient_graph_pcc(X, y, percentile=90):
    """
    Build patient graph using Pearson Correlation Coefficient (PCC).
    
    Args:
        X: (n_patients, n_selected_genes) - Expression data for selected DEGs only
        y: (n_patients,) - Patient labels
        percentile: Threshold percentile for edge creation
    """
    X = np.asarray(X, dtype=np.float32)

    # Fix: force 1D vectors to be 2D
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    elif X.ndim != 2:
        raise ValueError(f"X must be 2D for correlation. Got shape {X.shape}")

    if isinstance(y, pd.Series):
        y = y.values
    y = np.asarray(y)

    if X.shape[0] != len(y):
        raise ValueError(f"Number of samples in X and y do not match. X.shape={X.shape}, y.shape={y.shape}")

    for i, row in enumerate(X):
        if not hasattr(row, "shape"):
            print(f"[ERROR] Row {i} is a {type(row)}: {row}")

    # More efficient correlation computation using numpy
    corr_matrix = np.corrcoef(X)
    
    # Handle NaN values (in case of constant genes)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    
    # Zero out diagonal
    np.fill_diagonal(corr_matrix, 0)
    
    # Use absolute correlation for thresholding (strong positive or negative correlations)
    abs_corr_matrix = np.abs(corr_matrix)
    threshold = np.percentile(abs_corr_matrix, percentile)
    
    # Create edges based on threshold
    edge_indices = np.where(abs_corr_matrix >= threshold)
    edge_index = torch.tensor(np.stack(edge_indices), dtype=torch.long)

    edge_weights = abs_corr_matrix[edge_indices]
    edge_weight = torch.tensor(edge_weights, dtype=torch.float32)
    
    # Convert to tensors
    x = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    return Data(x=x, edge_index=edge_index, y=y_tensor, edge_weight=edge_weight)
class GCNModel(nn.Module):
    def __init__(self, num_features, hidden_channels=64, dropout=0.4):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.conv3 = GCNConv(hidden_channels, 2)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index, edge_weight)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index, edge_weight)
        return x
def train_gcn_single_split(X_train, y_train, X_test, y_test, epochs=200, hidden_channels = 64, dropout_rate=0.4, learning_rate = 0.01, weight_decay=1e-3, device='cpu'):
    """
    Train GCN with single train/test split for patient classification.
    
    Args:
        X_train: (n_train_patients, n_selected_genes) - Training data
        y_train: (n_train_patients,) - Training labels  
        X_test: (n_test_patients, n_selected_genes) - Test data
        y_test: (n_test_patients,) - Test labels
        epochs: Number of training epochs
        device: Training device
    """
    
    def init_weights(m):
        if hasattr(m, 'lin') and hasattr(m.lin, 'weight'):
            torch.nn.init.xavier_uniform_(m.lin.weight)
 
    # Combine for full graph construction
    X_combined = np.vstack([X_train, X_test])
    y_combined = np.concatenate([y_train, y_test])

    # Build patient graph using PCC
    graph = build_patient_graph_pcc(X_combined, y_combined).to(device)
    
    # Create train/test masks
    n_train = len(y_train)
    mask_train = torch.zeros(len(y_combined), dtype=torch.bool, device=device)
    mask_test = torch.zeros(len(y_combined), dtype=torch.bool, device=device)
    mask_train[:n_train] = True
    mask_test[n_train:] = True
    
    # Initialize model
    model = GCNModel(num_features=X_combined.shape[1], hidden_channels=hidden_channels, dropout=dropout_rate).to(device)
    model.apply(init_weights)
    
    # Class weights for imbalanced data
    weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Training tracking
    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []
    
    print(f"Training on {len(y_train)} samples, testing on {len(y_test)} samples")
    print(f"Graph has {graph.x.shape[0]} nodes and {graph.edge_index.shape[1]} edges")
    st.info(f"Training model for {epochs} epochs...")
        
    start_time = time.time()
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        optimizer.zero_grad()
        out = model(graph.x, graph.edge_index, graph.edge_weight)
        loss = criterion(out[mask_train], graph.y[mask_train])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        
        train_losses.append(loss.item())
        train_preds = out[mask_train].argmax(dim=1).cpu().numpy()
        train_acc = np.mean(train_preds == graph.y[mask_train].cpu().numpy())
        train_accuracies.append(train_acc)
        
        # Evaluation phase
        model.eval()
        with torch.no_grad():
            out = model(graph.x, graph.edge_index, graph.edge_weight)
            test_loss = criterion(out[mask_test], graph.y[mask_test]).item()
            test_losses.append(test_loss)
            test_preds = out[mask_test].argmax(dim=1).cpu().numpy()
            test_acc = np.mean(test_preds == graph.y[mask_test].cpu().numpy())
            test_accuracies.append(test_acc)
        
        scheduler.step(test_loss)
        
        # Print progress every 50 epochs
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs}: Train Loss: {loss.item():.4f}, "
                  f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
    end_time = time.time()
    elapsed_time = end_time - start_time

    st.success("✅ Training complete.")
    st.write(f"⏱️ Elapsed training time: **{elapsed_time:.2f} seconds**")
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        out = model(graph.x, graph.edge_index, graph.edge_weight)
        y_pred_final = out[mask_test].argmax(dim=1).cpu().numpy()
    
    y_true_final = y_test
    
    # Classification report
    report = classification_report(
        y_true_final, y_pred_final, 
        target_names=["Normal", "Cancer"], 
        output_dict=True
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true_final, y_pred_final)
    
    # Print results
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(classification_report(y_true_final, y_pred_final, target_names=["Normal", "Cancer"]))
    
    # Visualization
    # --- Confusion Matrix ---
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Cancer"],
                yticklabels=["Normal", "Cancer"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()
    
    # --- Loss Curves ---
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss', alpha=0.7)
    plt.plot(test_losses, label='Test Loss', alpha=0.7)
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # --- Accuracy Curves ---
    plt.figure(figsize=(8, 5))
    plt.plot(train_accuracies, label='Train Accuracy', alpha=0.7)
    plt.plot(test_accuracies, label='Test Accuracy', alpha=0.7)
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return report, cm, model, graph, mask_test, train_losses, test_losses, train_accuracies, test_accuracies

def compute_gene_stats(X, y, return_raw=False, add_features=False):
    """
    Compute log2FC and p-values for each gene between two classes.
    
    Args:
        X: (n_samples, n_genes) DataFrame of gene expression.
        y: (n_samples,) array-like of binary labels (0 or 1).
        return_raw: if True, returns (log2fc, pvals, gene_names)
        add_features: if True, returns additional gene-level stats

    Returns:
        If return_raw: log2fc, pvals, gene_names
        Else:
            - features: array of shape (n_genes, n_features)
            - gene_names: list of valid gene names
    """

    X0 = X[y == 0]
    X1 = X[y == 1]

    X0 = X0.astype(float)
    X1 = X1.astype(float)

    mean0 = X0.mean(axis=0)
    mean1 = X1.mean(axis=0)
    log2fc = np.log2((mean1 + 1e-8) / (mean0 + 1e-8))

    ttest_res = ttest_ind(X0, X1, axis=0, equal_var=False, nan_policy='omit')
    pvals = ttest_res.pvalue

    valid = (~np.isnan(log2fc)) & (~np.isnan(pvals))
    if valid.sum() == 0:
        raise ValueError("No valid genes found after filtering.")

    log2fc = log2fc[valid]
    pvals = pvals[valid]
    log10pval = -np.log10(np.clip(pvals, 1e-10, 1.0))
    gene_names = X.columns[valid]

    if return_raw:
        return log2fc, pvals, gene_names

    if add_features:
        from scipy.stats import skew, kurtosis
        mean_expr = X.mean(axis=0).values[valid]
        var_expr = X.var(axis=0).values[valid]
        skewness = skew(X.values, axis=0, nan_policy='omit')[valid]
        kurt = kurtosis(X.values, axis=0, nan_policy='omit')[valid]
        features = np.stack([np.abs(log2fc), log10pval, mean_expr, var_expr, skewness, kurt], axis=1)
    else:
        features = np.stack([np.abs(log2fc), log10pval], axis=1)

    features = np.nan_to_num(features)
    return features, gene_names

# -------------------------------
# 1. Load patient expression data
# -------------------------------
def step3_train_model():
    st.set_page_config(page_title="GEM-GRAPH App: Model Training", layout="wide")
    st.title("Step 3: Train GNN Model")
    if 'X_reduced' not in st.session_state or 'y' not in st.session_state:
        st.error("❌ Please complete Feature Selection first.")
        return

    X = st.session_state['X_reduced']
    y = np.array(st.session_state['y'])

        # Make sure y is int (not float)
    # if y.dtype == float:
    #     st.warning("⚠️ Target variable is float. Converting to int...")
    #     y = y.astype(int)

    # -------------------------------
    # 2. Select DEGs using volcano filtering
    # -------------------------------
    st.write("🔬 Generating DEG pseudo-labels using statistical thresholds...")
    stats_feat, pseudo_labels, gene_names, fc_threshold, pval_threshold = generate_pseudo_labels(
        X, y, fc_percentile=85, pval_percentile=15
    )

    # Log fold change & p-value (raw) for up/down separation
    log2fc, pvals, _ = compute_gene_stats(X, y, return_raw=True)
    deg_mask = pseudo_labels == 1
    deg_log2fc = log2fc[deg_mask]
    selected_deg_genes = gene_names[deg_mask]

    upregulated = selected_deg_genes[deg_log2fc > 0]
    downregulated = selected_deg_genes[deg_log2fc < 0]

    st.write(f"📌 Selected {len(selected_deg_genes)} DEGs")
    st.write(f"   📈 Upregulated genes ({len(upregulated)}): {list(upregulated)}")
    st.write(f"   📉 Downregulated genes ({len(downregulated)}): {list(downregulated)}")

    # -------------------------------
    # 3. Extract DEG subset for classification
    # -------------------------------
    X_deg = X_deg = X.loc[:, selected_deg_genes].values.astype(np.float32)



    # -------------------------------
    # 4. Train GCN on patient-level graph
    # -------------------------------
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train, X_test, y_train, y_test = train_test_split(
        X_deg, y, test_size=0.2, stratify=y, random_state=seed
    )

    out_channels = 2
    hidden_channels = st.slider("Hidden Channels", min_value=16, max_value=512, step=16, value=128)
    dropout_rate = st.slider("Dropout Rate", min_value=0.0, max_value=0.9, step=0.05, value=0.3)
    learning_rate = st.number_input("Learning Rate", min_value=1e-5, max_value=1e-1, value=1e-3, format="%.5f")
    weight_decay = st.number_input("Weight Decay", min_value=0.0, max_value=1e-1, value=5e-4, format="%.5f")
    epochs = st.slider("Epochs", min_value=10, max_value=500, step=10, value=100)
    threshold = st.slider("Graph Edge Threshold (correlation)", min_value=0.0, max_value=1.0, step=0.01, value=0.45)

    report, cm, model, graph, mask_test, train_losses, test_losses, train_accuracies, test_accuracies = train_gcn_single_split(
        X_train, y_train, X_test, y_test,
        epochs=epochs,
        hidden_channels=hidden_channels,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    model.eval()
    with torch.no_grad():
        out = model(graph.x, graph.edge_index, graph.edge_weight)
        y_pred = out[mask_test].argmax(dim=1).cpu().numpy()
        y_true = np.array(y_test)
        st.session_state["gcn_metrics"] = {
            "Model": "GCN",
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, average='weighted'),
            "Recall": recall_score(y_true, y_pred, average='weighted'),
            "F1-Score": f1_score(y_true, y_pred, average='weighted')
        }

    # -------------------------------
    # 5. Output report and save DEG list
    # -------------------------------

    st.write("\n🧾 Classification Report:")
    st.text(classification_report(y_true, y_pred, target_names=["Normal", "Cancer"]))

    st.write("\n✅ Final DEG gene list used in classification:")
    st.write(list(selected_deg_genes))

    # Confusion Matrix Plot
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Cancer"],
                yticklabels=["Normal", "Cancer"],
                ax=ax_cm)
    ax_cm.set_title("Confusion Matrix")
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("True")
    st.pyplot(fig_cm)

    # Loss Curve Plot
    fig_loss, ax_loss = plt.subplots(figsize=(8, 5))
    ax_loss.plot(train_losses, label='Train Loss', alpha=0.7)
    ax_loss.plot(test_losses, label='Test Loss', alpha=0.7)
    ax_loss.set_title("Loss Curves")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)
    st.pyplot(fig_loss)

    # Accuracy Curve Plot
    fig_acc, ax_acc = plt.subplots(figsize=(8, 5))
    ax_acc.plot(train_accuracies, label='Train Accuracy', alpha=0.7)
    ax_acc.plot(test_accuracies, label='Test Accuracy', alpha=0.7)
    ax_acc.set_title("Accuracy Curves")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)
    st.pyplot(fig_acc)

# import streamlit as st
# import numpy as np
# import torch
# import time
# from sklearn.model_selection import train_test_split
# from collections import Counter

# from utils.model_training import GCNModel, GATModel, build_gene_graph_rbf, train_model, evaluate_model, visualize_results
# from utils.smote_utils import apply_smote
# from utils.visualization import plot_losses, plot_confusion_matrix
# from sklearn.metrics import accuracy_score

# def step3_train_model():
#     st.title("Step 3: Train GNN Model")

#     if 'X_reduced' not in st.session_state or 'y' not in st.session_state:
#         st.error("❌ Please complete Feature Selection first.")
#         return

#     X = np.array(st.session_state['X_reduced'])
#     y = np.array(st.session_state['y'])

#     # Make sure y is int (not float)
#     if y.dtype == float:
#         st.warning("⚠️ Target variable is float. Converting to int...")
#         y = y.astype(int)

#     apply_smote_option = st.radio(
#         "Apply SMOTE to balance classes?",
#         options=["Yes", "No"]
#     )

#     if apply_smote_option == "Yes":
#         X, y = apply_smote(X, y)
#         st.success(f"✅ SMOTE applied. New class distribution: {Counter(y)}")
#     else:
#         st.info(f"ℹ️ SMOTE not applied. Class distribution: {Counter(y)}")

#     # ⚡ Regardless of SMOTE, always do this after split
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.4, random_state=42, stratify=y
#     )

#     # 🚨 Force X to float32 here
#     X_train = X_train.astype(np.float32)
#     X_test = X_test.astype(np.float32)

#     train_graph = build_gene_graph_rbf(X_train, y_train)
#     test_graph = build_gene_graph_rbf(X_test, y_test)

#     model_choice = st.radio(
#         "Select GNN Model:",
#         options=["GCN", "GAT"]
#     )

#     in_channels = train_graph.x.shape[1]
#     out_channels = 2
#     hidden_channels = st.slider("Hidden Channels", min_value=16, max_value=512, step=16, value=128)
#     dropout_rate = st.slider("Dropout Rate", min_value=0.0, max_value=0.9, step=0.05, value=0.3)
#     learning_rate = st.number_input("Learning Rate", min_value=1e-5, max_value=1e-1, value=1e-3, format="%.5f")
#     weight_decay = st.number_input("Weight Decay", min_value=0.0, max_value=1e-1, value=5e-4, format="%.5f")
#     epochs = st.slider("Epochs", min_value=10, max_value=500, step=10, value=100)
#     threshold = st.slider("Graph Edge Threshold (correlation)", min_value=0.0, max_value=1.0, step=0.01, value=0.45)

#     # Only show 'heads' when GAT is selected
#     if model_choice == "GAT":
#         heads = st.slider("Number of Attention Heads (GAT only)", min_value=1, max_value=16, step=1, value=4)
#     else:
#         heads = None  # Not applicable

#     if model_choice == "GCN":
#         model = GCNModel(num_features=in_channels, hidden_channels=hidden_channels, dropout=dropout_rate)
        
#     else:
#         model = GATModel(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels, heads=heads, dropout=dropout_rate)
    
#     optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

#     criterion = torch.nn.CrossEntropyLoss()

#     epochs = st.slider("Select number of epochs:", min_value=50, max_value=300, step=10, value=100)
#     st.info(f"Training model for {epochs} epochs...")
    
#     start_time = time.time()
#     train_losses, test_losses, train_accuracies, test_accuracies = train_model(
#         train_graph, test_graph, model, optimizer, criterion, epochs
#     )
#     end_time = time.time()
#     elapsed_time = end_time - start_time

#     st.success("✅ Training complete.")
#     st.write(f"⏱️ Elapsed training time: **{elapsed_time:.2f} seconds**")

#     y_true, y_pred = evaluate_model(test_graph, model)
#     acc = accuracy_score(y_true, y_pred)

#     st.write(f"🎯 Test Accuracy: **{acc:.4f}**")

#     visualize_results(
#         train_losses=train_losses,
#         test_losses=test_losses,
#         train_accuracies=[],  
#         test_accuracies=[],   
#         y_true=y_true,
#         y_pred=y_pred,
#         title_suffix=model_choice
#     )
