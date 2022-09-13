import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

class SVM:
    def __init__(self, kernel='linear', C=1.0, max_iter=1000, learning_rate=0.001, batch_size=1000):
        self.kernel = kernel
        self.C = C
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.w = None
        self.b = 0
        self.support_vectors = None
        self.support_vector_labels = None
        self.alpha = None

    def linear_kernel_batch(self, x1, x2):
        """Compute linear kernel for batches of data."""
        return np.dot(x1, x2.T)

    def rbf_kernel_batch(self, x1, x2, gamma=0.1):
        """Compute RBF kernel for batches of data."""
        norm_sq = np.sum(x1**2, axis=1).reshape(-1, 1) + \
                 np.sum(x2**2, axis=1) - \
                 2 * np.dot(x1, x2.T)
        return np.exp(-gamma * norm_sq)

    def compute_kernel_batch(self, x1, x2):
        """Compute kernel for a batch of data."""
        if self.kernel == 'linear':
            return self.linear_kernel_batch(x1, x2)
        return self.rbf_kernel_batch(x1, x2)

    def fit(self, X, y):
        """Train SVM using mini-batch processing."""
        n_samples, n_features = X.shape
        self.alpha = np.zeros(n_samples)
        
        # For linear SVM, initialize weights
        if self.kernel == 'linear':
            self.w = np.zeros(n_features)

        n_batches = int(np.ceil(n_samples / self.batch_size))
        
        for epoch in tqdm(range(self.max_iter), desc="Training SVM"):
            indices = np.random.permutation(n_samples)
            alpha_diff_total = 0
            
            for i in range(n_batches):
                start_idx = i * self.batch_size
                end_idx = min(start_idx + self.batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]
                
                # Compute kernel matrix for batch
                K_batch = self.compute_kernel_batch(X_batch, X_batch)
                
                # Update alpha for batch
                for j in range(len(batch_indices)):
                    idx = batch_indices[j]
                    if self.kernel == 'linear':
                        output = np.dot(X[idx], self.w) + self.b
                    else:
                        kernel_row = self.compute_kernel_batch(X[idx].reshape(1, -1), X_batch)
                        output = np.sum(self.alpha[batch_indices] * y_batch * kernel_row) + self.b
                    
                    error = output - y[idx]
                    alpha_old = self.alpha[idx]
                    self.alpha[idx] = np.clip(alpha_old - self.learning_rate * error * y[idx], 0, self.C)
                    
                    # Update weights for linear SVM
                    if self.kernel == 'linear':
                        self.w += (self.alpha[idx] - alpha_old) * y[idx] * X[idx]
                    
                    alpha_diff_total += abs(self.alpha[idx] - alpha_old)
                    
                    # Update bias
                    self.b -= self.learning_rate * error
            
            # Check convergence
            if alpha_diff_total < 1e-5:
                print(f"Converged at epoch {epoch}")
                break
        
        # Store support vectors (only store indices to save memory)
        sv_indices = self.alpha > 1e-5
        self.support_vectors = X[sv_indices]
        self.support_vector_labels = y[sv_indices]
        self.alpha = self.alpha[sv_indices]

    def predict(self, X):
        """Make predictions using batch processing."""
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)
        
        for i in range(0, n_samples, self.batch_size):
            end_idx = min(i + self.batch_size, n_samples)
            X_batch = X[i:end_idx]
            
            if self.kernel == 'linear':
                predictions[i:end_idx] = np.dot(X_batch, self.w) + self.b
            else:
                kernel_batch = self.compute_kernel_batch(X_batch, self.support_vectors)
                predictions[i:end_idx] = np.sum(self.alpha * 
                                              self.support_vector_labels * 
                                              kernel_batch, axis=1) + self.b
        
        return np.sign(predictions)

def generate_dataset(n_samples=100000, n_features=2, kernel_type='linear'):
    """Generate synthetic dataset with controlled size."""
    np.random.seed(42)
    
    if kernel_type == 'linear':
        X = np.random.randn(n_samples, n_features)
        w = np.random.randn(n_features)
        y = np.sign(np.dot(X, w) + np.random.normal(0, 0.1, n_samples))
    else:
        X = np.random.randn(n_samples, n_features)
        radius = np.sum(X**2, axis=1)
        y = np.where(radius < 2, 1, -1)
        noise_idx = np.random.choice(n_samples, size=int(0.1 * n_samples))
        y[noise_idx] *= -1
    
    return X, y

# Generate smaller test datasets first
n_samples = 10000  # Starting with a smaller dataset for testing
X_linear, y_linear = generate_dataset(n_samples=n_samples, kernel_type='linear')
X_nonlinear, y_nonlinear = generate_dataset(n_samples=n_samples, kernel_type='nonlinear')

# Standardize features
scaler = StandardScaler()
X_linear_scaled = scaler.fit_transform(X_linear)
X_nonlinear_scaled = scaler.fit_transform(X_nonlinear)

# Train and evaluate linear SVM
print("\nTraining Linear SVM...")
svm_linear = SVM(kernel='linear', C=1.0, max_iter=100, batch_size=1000)
svm_linear.fit(X_linear_scaled, y_linear)
y_pred_linear = svm_linear.predict(X_linear_scaled)
print("\nLinear SVM Accuracy:", accuracy_score(y_linear, y_pred_linear))
print("\nClassification Report:")
print(classification_report(y_linear, y_pred_linear))

# Train and evaluate RBF kernel SVM
print("\nTraining RBF Kernel SVM...")
svm_rbf = SVM(kernel='rbf', C=1.0, max_iter=100, batch_size=1000)
svm_rbf.fit(X_nonlinear_scaled, y_nonlinear)
y_pred_rbf = svm_rbf.predict(X_nonlinear_scaled)
print("\nRBF Kernel SVM Accuracy:", accuracy_score(y_nonlinear, y_pred_rbf))
print("\nClassification Report:")
print(classification_report(y_nonlinear, y_pred_rbf))

# Visualization function
def plot_decision_boundary(X, y, model, title):
    plt.figure(figsize=(10, 8))
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# Plot decision boundaries for a subset of data
plot_decision_boundary(X_linear_scaled[:1000], y_linear[:1000], 
                      svm_linear, 'Linear SVM Decision Boundary')
plot_decision_boundary(X_nonlinear_scaled[:1000], y_nonlinear[:1000], 
                      svm_rbf, 'RBF Kernel SVM Decision Boundary')