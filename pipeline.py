# --- IMPORTS ---
import numpy as np
import pandas as pd
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report


# --- CORE FUNCTIONS ---

def set_seed(seed=42):
    """Sets the random seed for reproducibility across multiple libraries."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_sample_graph(expression_df, selected_genes, label_col='label', threshold=0.7, scaler=None):
    """
    Constructs a graph from gene expression data.
    - Nodes are samples.
    - Edges are based on the correlation of gene expression profiles.
    - Returns edge_index, features, and labels for PyTorch Geometric.
    """
    filtered_df = expression_df[selected_genes + [label_col]]
    labels = torch.tensor(filtered_df[label_col].values, dtype=torch.long)
    features_df = filtered_df.drop(columns=[label_col])

    if scaler is None:
        scaler = StandardScaler()
        features = scaler.fit_transform(features_df.values)
    else:
        features = scaler.transform(features_df.values)

    corr = np.corrcoef(features)
    adj = (np.abs(corr) > threshold).astype(int)
    np.fill_diagonal(adj, 0)
    edge_index = np.array(np.nonzero(adj))

    return torch.tensor(edge_index, dtype=torch.long), torch.tensor(features, dtype=torch.float), labels, scaler


class GCNModel(nn.Module):
    """
    A simple Graph Convolutional Network model for classification.
    - Uses three GCN layers with ReLU activation and dropout.
    - Includes BatchNorm1d for improved training stability.
    """

    def __init__(self, num_features, num_classes, hidden_channels=64, dropout=0.4):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.conv3 = GCNConv(hidden_channels, num_classes)
        self.dropout = dropout
        self.apply(self.init_weights)

    def init_weights(self, m):
        """Initializes the weights of linear layers using Xavier uniform initialization."""
        if hasattr(m, 'lin') and hasattr(m.lin, 'weight'):
            torch.nn.init.xavier_uniform_(m.lin.weight)

    def forward(self, x, edge_index, edge_weight=None):
        """Forward pass through the GCN layers."""
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


def calculate_integrated_gradients(model, x, edge_index, target_label, num_steps=50):
    """
    Calculates Integrated Gradients for feature importance.
    - Measures the attribution of each gene (feature) to the model's prediction
      for a specific target class.
    """
    model.eval()
    baseline = torch.zeros_like(x)
    importance_scores = torch.zeros_like(x)
    for i in range(num_steps + 1):
        alpha = i / num_steps
        interpolated_input = baseline + alpha * (x - baseline)
        interpolated_input.requires_grad = True
        output = model(interpolated_input, edge_index)
        target_output = output[:, target_label].sum()
        model.zero_grad()
        target_output.backward()
        if interpolated_input.grad is not None:
            importance_scores += interpolated_input.grad.detach()
    integrated_gradients = (x - baseline) * importance_scores / num_steps
    gene_importance_scores = torch.abs(integrated_gradients).sum(dim=0).cpu().numpy()
    return gene_importance_scores


def recursive_gene_elimination(expression_df, device, hidden_channels, epochs, elimination_rate, min_genes_to_keep,
                               seed, label_col, _status_placeholder, progress_bar):
    """
    Performs recursive gene elimination based on Integrated Gradients.
    - Iteratively trains a GCN model and removes the least important genes.
    """
    set_seed(seed)

    current_genes = expression_df.drop(columns=label_col).columns.tolist()
    best_genes = current_genes.copy()
    best_val_accuracy = 0.0
    best_genes_ig_scores = None

    if min_genes_to_keep is None:
        min_genes_to_keep = 1

    # Split data for this inner loop
    train_val_df, _ = train_test_split(expression_df, test_size=0.2, random_state=42, stratify=expression_df[label_col])
    train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=42, stratify=train_val_df[label_col])

    total_steps = len(current_genes)
    step_count = 0

    while len(current_genes) > min_genes_to_keep:
        # Update progress bar and status message
        progress = step_count / total_steps
        progress_bar.progress(progress)

        train_df_subset = train_df[current_genes + [label_col]]
        val_df_subset = val_df[current_genes + [label_col]]

        edge_index_train, x_train, y_train, scaler = build_sample_graph(train_df_subset, current_genes,
                                                                        label_col=label_col)
        edge_index_val, x_val, y_val, _ = build_sample_graph(val_df_subset, current_genes, label_col=label_col,
                                                             scaler=scaler)

        num_samples_train, num_genes = x_train.shape
        num_classes = len(y_train.unique())

        x_train, y_train, edge_index_train = x_train.to(device), y_train.to(device), edge_index_train.to(device)
        x_val, y_val, edge_index_val = x_val.to(device), y_val.to(device), edge_index_val.to(device)

        y_train_np = y_train.cpu().numpy()
        weights = compute_class_weight('balanced', classes=np.unique(y_train_np), y=y_train_np)

        model = GCNModel(num_features=num_genes, num_classes=num_classes, hidden_channels=hidden_channels).to(device)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-3)

        # Training loop
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            out = model(x_train, edge_index_train)
            train_loss = criterion(out, y_train)
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

        val_acc = \
        evaluate_epoch(model, x_val, y_val, edge_index_val, torch.ones(x_val.shape[0], dtype=torch.bool, device=device),
                       criterion)[1]
        _status_placeholder.write(f"Number of genes: {len(current_genes):<4}, Final Validation Accuracy: {val_acc:.4f}")

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_genes = current_genes.copy()
            # Capture the scores from the best run
            combined_scores = np.zeros(num_genes)
            for i_class in range(num_classes):
                ig_scores = calculate_integrated_gradients(model, x_train, edge_index_train, target_label=i_class)
                combined_scores += np.abs(ig_scores)
            best_genes_ig_scores = combined_scores
        elif val_acc == best_val_accuracy and len(current_genes) < len(best_genes):
            best_genes = current_genes.copy()
            # Recapture scores for the new best run
            combined_scores = np.zeros(num_genes)
            for i_class in range(num_classes):
                ig_scores = calculate_integrated_gradients(model, x_train, edge_index_train, target_label=i_class)
                combined_scores += np.abs(ig_scores)
            best_genes_ig_scores = combined_scores

        combined_scores = np.zeros(num_genes)
        for i_class in range(num_classes):
            ig_scores = calculate_integrated_gradients(model, x_train, edge_index_train, target_label=i_class)
            combined_scores += np.abs(ig_scores)

        ranked_genes = pd.DataFrame({'Gene': current_genes, 'ImportanceScore': combined_scores}).sort_values(
            by='ImportanceScore', ascending=False)
        elimination_count = max(1, int(len(current_genes) * elimination_rate))
        if len(current_genes) - elimination_count < min_genes_to_keep:
            elimination_count = len(current_genes) - min_genes_to_keep
        if elimination_count <= 0:
            break

        genes_to_eliminate = ranked_genes['Gene'].tail(elimination_count).tolist()
        current_genes = [g for g in current_genes if g not in genes_to_eliminate]
        step_count += elimination_count

    return best_val_accuracy, len(best_genes), best_genes, best_genes_ig_scores


def validate_gene_set(expression_df, selected_genes, device, hidden_channels, epochs, seed=42, label_col='label'):
    """
    Validates the selected gene set on a held-out test set.
    """
    set_seed(seed)
    train_df, test_df = train_test_split(expression_df, test_size=0.3, random_state=seed,
                                         stratify=expression_df[label_col])
    train_df_subset = train_df[selected_genes + [label_col]]
    test_df_subset = test_df[selected_genes + [label_col]]

    edge_index_train, x_train, y_train, scaler = build_sample_graph(train_df_subset, selected_genes,
                                                                    label_col=label_col)
    num_samples_train, num_features = x_train.shape
    num_classes = len(y_train.unique())

    x_train, y_train, edge_index_train = x_train.to(device), y_train.to(device), edge_index_train.to(device)
    y_train_np = y_train.cpu().numpy()
    weights = compute_class_weight('balanced', classes=np.unique(y_train_np), y=y_train_np)

    model = GCNModel(num_features, num_classes, hidden_channels=hidden_channels).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-3)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(x_train, edge_index_train)
        loss = criterion(out, y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

    edge_index_test, x_test, y_test, _ = build_sample_graph(test_df_subset, selected_genes, label_col=label_col,
                                                            scaler=scaler)
    x_test, y_test, edge_index_test = x_test.to(device), y_test.to(device), edge_index_test.to(device)

    model.eval()
    with torch.no_grad():
        out = model(x_test, edge_index_test)
        pred = out.argmax(dim=1)

    y_test_np = y_test.cpu().numpy()
    pred_np = pred.cpu().numpy()
    accuracy = (pred == y_test).float().mean().item()

    return accuracy, y_test_np, pred_np


def evaluate_epoch(model, x, y, edge_index, mask, criterion):
    """Evaluates the model on a given dataset split."""
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index)
        loss = criterion(out[mask], y[mask]).item()
        pred = out[mask].argmax(dim=1)
        accuracy = (pred == y[mask]).float().mean().item()
    return loss, accuracy
