import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Create a synthetic dataset
def generate_data(n_samples=1000, n_features=5):
    np.random.seed(42)  # For reproducibility
    # Generate random data
    X = np.random.rand(n_samples, n_features)
    # Introduce correlation between features
    X[:, 1] = X[:, 0] + np.random.normal(0, 0.1, n_samples)  # Correlated with feature 0
    X[:, 2] = X[:, 0] + np.random.normal(0, 0.1, n_samples)  # Correlated with feature 0
    X[:, 3] = X[:, 1] + np.random.normal(0, 0.1, n_samples)  # Correlated with feature 1
    X[:, 4] = X[:, 2] + np.random.normal(0, 0.1, n_samples)  # Correlated with feature 2
    return X

# Step 2: Implement PCA from Scratch
class PCA:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.eigenvectors = None
        self.eigenvalues = None
        self.mean = None

    def fit(self, X):
        # Step 1: Standardize the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # Step 2: Compute the covariance matrix
        covariance_matrix = np.cov(X_centered, rowvar=False)
        
        # Step 3: Compute eigenvalues and eigenvectors
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(covariance_matrix)
        
        # Step 4: Sort the eigenvectors by eigenvalues in descending order
        sorted_indices = np.argsort(self.eigenvalues)[::-1]
        self.eigenvalues = self.eigenvalues[sorted_indices]
        self.eigenvectors = self.eigenvectors[:, sorted_indices]
        
        # Step 5: Select the top n_components eigenvectors
        self.eigenvectors = self.eigenvectors[:, :self.n_components]

    def transform(self, X):
        # Center the data
        X_centered = X - self.mean
        # Project the data onto the selected eigenvectors
        return np.dot(X_centered, self.eigenvectors)

# Step 3: Visualize the Results
def visualize_results(original_data, reduced_data):
    plt.figure(figsize=(12, 6))
    
    # Original data
    plt.subplot(1, 2, 1)
    plt.scatter(original_data[:, 0], original_data[:, 1], alpha=0.5)
    plt.title("Original Data")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    
    # Reduced data
    plt.subplot(1, 2, 2)
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], alpha=0.5)
    plt.title("PCA Reduced Data")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    
    plt.tight_layout()
    plt.show()

# Create a synthetic dataset
data = generate_data(n_samples=1000, n_features=5)

# Apply PCA
pca = PCA(n_components=2)
pca.fit(data)
reduced_data = pca.transform(data)

# Visualize the results
visualize_results(data, reduced_data)