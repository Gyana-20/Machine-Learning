import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm

class KNNClassifier:
    def __init__(self, k=5, metric='euclidean', batch_size=1000):
        """
        Initialize KNN Classifier.
        
        Parameters:
        k (int): Number of neighbors to consider
        metric (str): Distance metric to use ('euclidean' or 'manhattan')
        batch_size (int): Batch size for memory-efficient prediction
        """
        self.k = k
        self.metric = metric
        self.batch_size = batch_size
        self.X_train = None
        self.y_train = None
        
    def euclidean_distance(self, x1, x2):
        """Calculate Euclidean distance between points."""
        return np.sqrt(np.sum((x1 - x2) ** 2, axis=1))
    
    def manhattan_distance(self, x1, x2):
        """Calculate Manhattan distance between points."""
        return np.sum(np.abs(x1 - x2), axis=1)
    
    def fit(self, X, y):
        """
        Store training data.
        
        Parameters:
        X (array-like): Training features
        y (array-like): Training labels
        """
        self.X_train = X
        self.y_train = y
        
    def predict_single(self, x):
        """
        Predict class for a single instance.
        
        Parameters:
        x (array-like): Instance to classify
        
        Returns:
        int: Predicted class
        """
        # Calculate distances
        if self.metric == 'euclidean':
            distances = self.euclidean_distance(self.X_train, x.reshape(1, -1))
        else:
            distances = self.manhattan_distance(self.X_train, x.reshape(1, -1))
        
        # Get k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        
        # Return most common class
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
    
    def predict(self, X):
        """
        Predict classes for multiple instances using batch processing.
        
        Parameters:
        X (array-like): Instances to classify
        
        Returns:
        array: Predicted classes
        """
        predictions = []
        
        # Use batch processing to handle large datasets
        for i in tqdm(range(0, len(X), self.batch_size), desc="Predicting"):
            batch = X[i:i + self.batch_size]
            batch_predictions = [self.predict_single(x) for x in batch]
            predictions.extend(batch_predictions)
            
        return np.array(predictions)
    
    def get_k_neighbors(self, x):
        """
        Get k nearest neighbors for a single instance.
        
        Parameters:
        x (array-like): Instance to find neighbors for
        
        Returns:
        tuple: (indices of neighbors, distances to neighbors)
        """
        if self.metric == 'euclidean':
            distances = self.euclidean_distance(self.X_train, x.reshape(1, -1))
        else:
            distances = self.manhattan_distance(self.X_train, x.reshape(1, -1))
            
        k_indices = np.argsort(distances)[:self.k]
        return k_indices, distances[k_indices]

def generate_dataset(n_samples=100000, n_features=2, n_classes=3):
    """
    Generate synthetic dataset suitable for KNN classification.
    
    Parameters:
    n_samples (int): Number of samples to generate
    n_features (int): Number of features
    n_classes (int): Number of classes
    
    Returns:
    tuple: (features array, labels array)
    """
    np.random.seed(42)
    
    # Generate cluster centers
    centers = np.random.randn(n_classes, n_features) * 5
    
    # Generate samples around centers
    X = []
    y = []
    samples_per_class = n_samples // n_classes
    
    for i in range(n_classes):
        # Generate samples for each class with varying variance
        cluster = np.random.randn(samples_per_class, n_features) * (i + 1) * 0.5
        cluster = cluster + centers[i]
        X.append(cluster)
        y.append(np.full(samples_per_class, i))
    
    X = np.vstack(X)
    y = np.hstack(y)
    
    # Shuffle the dataset
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    return X, y

# Generate synthetic dataset
print("Generating synthetic dataset...")
X, y = generate_dataset(n_samples=100000, n_features=2, n_classes=3)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into train and test sets
def train_test_split(X, y, test_size=0.2):
    """Split data into training and testing sets."""
    n_test = int(len(X) * test_size)
    indices = np.random.permutation(len(X))
    test_idx, train_idx = indices[:n_test], indices[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)

# Train and evaluate KNN
print("\nTraining and evaluating KNN classifier...")
knn = KNNClassifier(k=5, metric='euclidean', batch_size=1000)
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualization function
def plot_decision_boundary(X, y, model, title, plot_points=1000):
    """
    Plot decision boundary and data points.
    
    Parameters:
    X (array-like): Feature data
    y (array-like): Labels
    model: Trained model
    title (str): Plot title
    plot_points (int): Number of points to plot
    """
    plt.figure(figsize=(10, 8))
    
    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    # Predict for each point in the mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    plt.contourf(xx, yy, Z, alpha=0.4)
    
    # Plot a subset of points
    indices = np.random.choice(len(X), plot_points, replace=False)
    plt.scatter(X[indices, 0], X[indices, 1], c=y[indices], alpha=0.8)
    
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# Plot decision boundary with a subset of points
print("\nPlotting decision boundary...")
plot_decision_boundary(X_scaled[:1000], y[:1000], knn, 
                      f'KNN Decision Boundary (k={knn.k})')

# Additional analysis
def analyze_k_sensitivity(X_train, X_test, y_train, y_test, k_values):
    """Analyze model sensitivity to different k values."""
    accuracies = []
    for k in k_values:
        knn = KNNClassifier(k=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
    return accuracies

# Test different k values
k_values = [3, 5, 7, 9, 11]
accuracies = analyze_k_sensitivity(X_train, X_test, y_train, y_test, k_values)

# Plot k sensitivity
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o')
plt.title('Model Accuracy vs. k Value')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()