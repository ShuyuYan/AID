import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


class ClusterNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, out_dim=10):
        super(ClusterNet, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        return self.mlp(x)


def plot_clusters(X, labels, title="Cluster Visualization"):
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='viridis')
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.title(title)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    excel_path = "/home/yanshuyu/Data/AID/20260111_1044基线及用药.xlsx"
    start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    df = pd.read_excel(excel_path, sheet_name='cluster')
    X = df.select_dtypes(include=['int64', 'float64'])

    imputer = SimpleImputer(strategy="mean")
    X_np = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_np)
    input_dim = X_scaled.shape[1]
    model = ClusterNet(input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.MSELoss()

    num_epochs = 100
    batch_size = 8

    for epoch in range(num_epochs):
        model.train()
        permutation = torch.randperm(X_scaled.shape[0])

        epoch_loss = 0.0
        for i in range(0, X_scaled.shape[0], batch_size):
            optimizer.zero_grad()
            indices = permutation[i:i + batch_size]
            batch = torch.tensor(X_scaled[indices], dtype=torch.float32).to(device)
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / (X_scaled.shape[0] / batch_size):.4f}")

    model.eval()
    with torch.no_grad():
        features = model(torch.tensor(X_scaled, dtype=torch.float32).to(device))

    features_np = features.cpu().numpy()

    kmeans = KMeans(n_clusters=3, random_state=42)
    cluster_labels = kmeans.fit_predict(features_np)
    print(cluster_labels.sum())

    df['type'] = cluster_labels
    output_file = f"/home/yanshuyu/Data/AID/clusternet_output2.xlsx"
    df.to_excel(output_file, index=False)
    plot_clusters(features_np, cluster_labels, title="ClusterNet Clustering Results")