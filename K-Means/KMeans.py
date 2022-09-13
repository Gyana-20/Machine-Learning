import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score

# Step 1: Create a synthetic dataset
def generate_data(n_samples=1000, n_clusters=3):
    np.random.seed(42)  # For reproducibility
    # Generate data for each cluster
    clusters = []
    for i in range(n_clusters):
        mean = np.random.uniform(-10, 10, size=2)
        cov = np.eye(2)
        cluster = np.random.multivariate_normal(mean, cov, size=n_samples // n_clusters)
        clusters.append(cluster)
    # Combine the clusters into a single dataset
    data = np.vstack(clusters)
    # Create a target variable for the actual clusters
    target = np.repeat(range(n_clusters), n_samples // n_clusters)
    return data, target

# Step 2: Implement K-means Clustering from Scratch
class KMeans:
    def __init__(self, n_clusters=3, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None
        self.labels = None

    def _initialize_centroids(self, data):
        np.random.seed(42)  # For reproducibility
        indices = np.random.choice(data.shape[0], size=self.n_clusters, replace=False)
        self.centroids = data[indices]

    def _assign_clusters(self, data):
        distances = np.sqrt(((data - self.centroids[:, np.newaxis])**2).sum(axis=2))
        self.labels = np.argmin(distances, axis=0)

    def _update_centroids(self, data):
        self.centroids = np.array([data[self.labels == i].mean(axis=0) for i in range(self.n_clusters)])

    def fit(self, data):
        self._initialize_centroids(data)
        for _ in range(self.max_iter):
            previous_centroids = self.centroids
            self._assign_clusters(data)
            self._update_centroids(data)
            if np.all(self.centroids == previous_centroids):
                break

# Step 3: Evaluate the Model
def evaluate_model(data, target, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(data)
    accuracy = adjusted_rand_score(target, kmeans.labels)
    return accuracy

# Create a synthetic dataset
data, target = generate_data(n_samples=1000, n_clusters=3)

# Evaluate the model
accuracy = evaluate_model(data, target, n_clusters=3)
print(f"Adjusted Rand Index: {accuracy:.4f}")

# Visualize the clusters
plt.scatter(data[:, 0], data[:, 1], c=target)
plt.title("Actual Clusters")
plt.show()

kmeans = KMeans(n_clusters=3)
kmeans.fit(data)
plt.scatter(data[:, 0], data[:, 1], c=kmeans.labels)
plt.title("Assigned Clusters")
plt.show()