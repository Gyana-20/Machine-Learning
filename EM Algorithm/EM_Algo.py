import numpy as np

import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs


# Step 1: Create a synthetic dataset

def create_synthetic_data(n_samples=1000, n_features=2, n_clusters=2):

    X, _ = make_blobs(n_samples=n_samples, centers=n_clusters, n_features=n_features, random_state=42)

    return X


# Step 2: Implement the EM Algorithm for GMM

class GaussianMixtureModel:

    def __init__(self, n_components=2, max_iter=100, tol=1e-4):

        self.n_components = n_components

        self.max_iter = max_iter

        self.tol = tol


    def fit(self, X):

        n_samples, n_features = X.shape

        

        # Initialize parameters

        self.means = X[np.random.choice(n_samples, self.n_components, replace=False)]

        self.covariances = np.array([np.eye(n_features)] * self.n_components)

        self.weights = np.ones(self.n_components) / self.n_components

        

        log_likelihoods = []


        for _ in range(self.max_iter):

            # E-step

            responsibilities = self._e_step(X)

            

            # M-step

            self._m_step(X, responsibilities)

            

            # Check for convergence

            log_likelihood = self._compute_log_likelihood(X)

            log_likelihoods.append(log_likelihood)

            if len(log_likelihoods) > 1 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < self.tol:

                break


    def _e_step(self, X):

        """Calculate the responsibilities."""

        likelihoods = np.zeros((X.shape[0], self.n_components))

        for k in range(self.n_components):

            likelihoods[:, k] = self.weights[k] * self._multivariate_gaussian(X, self.means[k], self.covariances[k])

        responsibilities = likelihoods / likelihoods.sum(axis=1, keepdims=True)

        return responsibilities


    def _m_step(self, X, responsibilities):

        """Update the parameters."""

        n_samples, n_features = X.shape

        for k in range(self.n_components):

            N_k = responsibilities[:, k].sum()

            self.means[k] = (1 / N_k) * np.sum(responsibilities[:, k].reshape(-1, 1) * X, axis=0)

            diff = X - self.means[k]

            self.covariances[k] = (1 / N_k) * np.dot(responsibilities[:, k] * diff.T, diff)

            self.weights[k] = N_k / n_samples


    def _multivariate_gaussian(self, X, mean, covariance):

        """Calculate the multivariate Gaussian probability density function."""

        d = X.shape[1]

        coeff = 1 / np.sqrt((2 * np.pi) ** d * np.linalg.det(covariance))

        diff = X - mean

        exponent = -0.5 * np.sum(np.dot(diff, np.linalg.inv(covariance)) * diff, axis=1)

        return coeff * np.exp(exponent)


    def _compute_log_likelihood(self, X):

        """Compute the log likelihood of the data given the current parameters."""

        likelihoods = np.zeros((X.shape[0], self.n_components))

        for k in range(self.n_components):

            likelihoods[:, k] = self.weights[k] * self._multivariate_gaussian(X, self.means[k], self.covariances[k])

        return np.sum(np.log(likelihoods.sum(axis=1) + 1e-10))  # Avoid log(0)


    def predict(self, X):

        """Predict the cluster for each sample."""

        responsibilities = self._e_step(X)

        return np.argmax(responsibilities, axis=1)


# Step 3: Visual ize the Results

def visualize_results(X, gmm):

    plt.figure(figsize=(10, 6))

    plt.scatter(X[:, 0], X[:, 1], c=gmm.predict(X), cmap='viridis', s=30, marker='o', edgecolor='k')

    

    # Plot the Gaussian components

    for mean, cov in zip(gmm.means, gmm.covariances):

        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # Create an ellipse to represent the Gaussian component

        v = np.sqrt(2) * np.sqrt(eigenvalues)

        angle = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0])

        ell = plt.matplotlib.patches.Ellipse(mean, v[0], v[1], angle=np.degrees(angle), color='red', alpha=0.5)

        plt.gca().add_patch(ell)


    plt.title("Gaussian Mixture Model Clustering")

    plt.xlabel("Feature 1")

    plt.ylabel("Feature 2")

    plt.grid()

    plt.show()
    

# Create a synthetic dataset

X = create_synthetic_data(n_samples=1000, n_features=2, n_clusters=2)


# Fit the GMM model

gmm = GaussianMixtureModel(n_components=2)

gmm.fit(X)


# Visualize the results

visualize_results(X, gmm)

