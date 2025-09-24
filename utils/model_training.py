
# import streamlit as st
# from collections import Counter

# # Data Handling
# import numpy as np
# import pandas as pd

# # PyTorch + PyTorch Geometric
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.data import Data
# from torch_geometric.nn import GCNConv, GATConv

# # Scikit-learn
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# from sklearn.utils.class_weight import compute_class_weight
# from sklearn.metrics.pairwise import rbf_kernel

# # Statistical Testing
# from scipy.stats import ttest_ind, pearsonr

# # Visualization (optional, for loss curves or heatmaps)
# import matplotlib.pyplot as plt
# import seaborn as sns

# # 1. Build Graph
# def build_gene_graph_rbf(X_features, y_array, gamma=0.5, dynamic_percentile=85):
#     try:
#         sim_matrix = rbf_kernel(X_features, gamma=gamma)
#     except Exception as e:
#         print("🔥 RBF kernel error:", str(e))
#         raise

#     np.fill_diagonal(sim_matrix, 0)
#     threshold = np.percentile(sim_matrix, dynamic_percentile)

#     edge_index = []
#     for i in range(sim_matrix.shape[0]):
#         for j in range(i + 1, sim_matrix.shape[1]):
#             if sim_matrix[i, j] >= threshold:
#                 edge_index.append([i, j])
#                 edge_index.append([j, i])

#     edge_index = torch.tensor(edge_index, dtype=torch.long).T
#     x = torch.tensor(X_features, dtype=torch.float32)
#     y = torch.tensor(y_array, dtype=torch.long)

#     return Data(x=x, edge_index=edge_index, y=y)

# # 2. Balance Data + Create Graphs
# def balance_data_and_create_graphs(X, y, apply_smote=True, test_size=0.6, random_state=42):
#     y_int = y.astype(int)

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y_int, test_size=test_size,
#         random_state=random_state, stratify=y_int
#     )

#     if apply_smote:
#         smote = SMOTE(random_state=random_state)
#         X_train, y_train = smote.fit_resample(X_train, y_train)
#         st.success(f"✅ Applied SMOTE. New train distribution: {Counter(y_train)}")
#     else:
#         st.info(f"ℹ️ SMOTE not applied. Train distribution: {Counter(y_train)}")

#     train_graph = build_gene_graph_rbf(X_train, y_train)
#     test_graph = build_gene_graph_rbf(X_test, y_test)

#     return train_graph, test_graph

# # 3. GCN Model
# class GCNModel(nn.Module):
#     def __init__(self, num_features, hidden_channels=64, dropout=0.4):
#         super(GCNModel, self).__init__()
#         self.conv1 = GCNConv(num_features, hidden_channels)
#         self.bn1 = nn.BatchNorm1d(hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, hidden_channels)
#         self.bn2 = nn.BatchNorm1d(hidden_channels)
#         self.conv3 = GCNConv(hidden_channels, 2)
#         self.dropout = dropout

#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index)
#         x = self.bn1(x)
#         x = F.relu(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)

#         x = self.conv2(x, edge_index)
#         x = self.bn2(x)
#         x = F.relu(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)

#         x = self.conv3(x, edge_index)
#         return x

# # 4. GAT Model
# class GATModel(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.6):
#         super(GATModel, self).__init__()
#         self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
#         self.bn1 = torch.nn.BatchNorm1d(hidden_channels * heads)
#         self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1, dropout=dropout)
#         self.dropout = dropout

#     def forward(self, x, edge_index, edge_attr=None):
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.gat1(x, edge_index)
#         x = self.bn1(x)
#         x = F.relu(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.gat2(x, edge_index)
#         return F.log_softmax(x, dim=1)

# # 5. Training Function
# def train_model(train_data, test_data, model, optimizer, criterion, epochs=100):
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

#     train_losses = []
#     test_losses = []
#     train_accuracies = []
#     test_accuracies = []

#     for epoch in range(epochs):
#         model.train()
#         optimizer.zero_grad()
#         out = model(train_data.x, train_data.edge_index)
#         loss = criterion(out, train_data.y)
#         loss.backward()
#         optimizer.step()
#         scheduler.step()

#         train_losses.append(loss.item())
#         train_preds = out.argmax(dim=1)
#         train_acc = accuracy_score(train_data.y.cpu(), train_preds.cpu())
#         train_accuracies.append(train_acc)

#         model.eval()
#         with torch.no_grad():
#             test_out = model(test_data.x, test_data.edge_index)
#             test_loss = criterion(test_out, test_data.y)
#             test_preds = test_out.argmax(dim=1)
#             test_acc = accuracy_score(test_data.y.cpu(), test_preds.cpu())

#             test_losses.append(test_loss.item())
#             test_accuracies.append(test_acc)

#     return train_losses, test_losses, train_accuracies, test_accuracies



# # 6. Evaluate model
# def evaluate_model(data, model):
#     model.eval()
#     with torch.no_grad():
#         out = model(data.x, data.edge_index)
#         preds = out.argmax(dim=1)
#         return data.y.cpu().numpy(), preds.cpu().numpy()
    
# def visualize_results(train_losses, test_losses, train_accuracies, test_accuracies, y_true, y_pred, title_suffix="GCN"):
#     # --- Classification Report ---
#     st.subheader("📋 Classification Report")
#     report = classification_report(y_true, y_pred, target_names=["Non-DEG", "DEG"])
#     st.text(report)

#     # --- Confusion Matrix ---
#     cm = confusion_matrix(y_true, y_pred)
#     fig_cm, ax_cm = plt.subplots()
#     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-DEG", "DEG"], yticklabels=["Non-DEG", "DEG"], ax=ax_cm)
#     ax_cm.set_title(f"Confusion Matrix - {title_suffix}")
#     ax_cm.set_xlabel("Predicted")
#     ax_cm.set_ylabel("True")
#     st.subheader("📊 Confusion Matrix")
#     st.pyplot(fig_cm)

#     # --- Loss Plot ---
#     fig_loss, ax_loss = plt.subplots()
#     ax_loss.plot(train_losses, label="Train Loss")
#     ax_loss.plot(test_losses, label="Test Loss")
#     ax_loss.set_title(f"Loss per Epoch - {title_suffix}")
#     ax_loss.set_xlabel("Epoch")
#     ax_loss.set_ylabel("CrossEntropy Loss")
#     ax_loss.legend()
#     ax_loss.grid(True)
#     st.subheader("📉 Loss Curves")
#     st.pyplot(fig_loss)

#     # --- Accuracy Plot (only if data provided) ---
#     if train_accuracies and test_accuracies:
#         fig_acc, ax_acc = plt.subplots()
#         ax_acc.plot(train_accuracies, label="Train Accuracy")
#         ax_acc.plot(test_accuracies, label="Test Accuracy")
#         ax_acc.set_title(f"Accuracy per Epoch - {title_suffix}")
#         ax_acc.set_xlabel("Epoch")
#         ax_acc.set_ylabel("Accuracy")
#         ax_acc.legend()
#         ax_acc.grid(True)
#         st.subheader("📈 Accuracy Curves")
#         st.pyplot(fig_acc)