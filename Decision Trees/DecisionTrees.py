import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm

class Node:
    """Node class for Decision Tree."""
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature  # Feature index for splitting
        self.threshold = threshold  # Threshold value for splitting
        self.left = left  # Left child node
        self.right = right  # Right child node
        self.value = value  # Predicted class (for leaf nodes)

class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2, min_gain=1e-4):
        """
        Initialize Decision Tree Classifier.
        
        Parameters:
        max_depth (int): Maximum depth of the tree
        min_samples_split (int): Minimum samples required to split a node
        min_gain (float): Minimum information gain required to split
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.root = None
        self.n_classes = None
        self.feature_importances_ = None
    
    def entropy(self, y):
        """Calculate entropy of label distribution."""
        proportions = np.bincount(y) / len(y)
        entropy = -np.sum([p * np.log2(p) for p in proportions if p > 0])
        return entropy
    
    def information_gain(self, y, y_left, y_right):
        """Calculate information gain for a split."""
        p_left = len(y_left) / len(y)
        p_right = len(y_right) / len(y)
        return self.entropy(y) - (p_left * self.entropy(y_left) + p_right * self.entropy(y_right))
    
    def split_data(self, X, y, feature, threshold):
        """Split data based on feature and threshold."""
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        return X[left_mask], X[right_mask], y[left_mask], y[right_mask]
    
    def find_best_split(self, X, y):
        """Find the best split for a node."""
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        n_features = X.shape[1]
        
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                X_left, X_right, y_left, y_right = self.split_data(X, y, feature, threshold)
                
                if len(y_left) < self.min_samples_split or len(y_right) < self.min_samples_split:
                    continue
                
                gain = self.information_gain(y, y_left, y_right)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def build_tree(self, X, y, depth=0):
        """Recursively build the decision tree."""
        n_samples, n_features = X.shape
        
        # Check stopping criteria
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            len(np.unique(y)) == 1):
            leaf_value = np.argmax(np.bincount(y))
            return Node(value=leaf_value)
        
        # Find best split
        best_feature, best_threshold, best_gain = self.find_best_split(X, y)
        
        # Create leaf node if no good split is found
        if best_gain < self.min_gain:
            leaf_value = np.argmax(np.bincount(y))
            return Node(value=leaf_value)
        
        # Split data and create child nodes
        X_left, X_right, y_left, y_right = self.split_data(X, y, best_feature, best_threshold)
        
        left_subtree = self.build_tree(X_left, y_left, depth + 1)
        right_subtree = self.build_tree(X_right, y_right, depth + 1)
        
        return Node(best_feature, best_threshold, left_subtree, right_subtree)
    
    def fit(self, X, y):
        """
        Train the decision tree.
        
        Parameters:
        X (array-like): Training features
        y (array-like): Training labels
        """
        self.n_classes = len(np.unique(y))
        self.n_features = X.shape[1]
        self.feature_importances_ = np.zeros(self.n_features)
        self.root = self.build_tree(X, y)
        
        # Calculate feature importances
        self._calculate_feature_importances(self.root, 1.0)
        self.feature_importances_ /= np.sum(self.feature_importances_)
        
        return self
    
    def _calculate_feature_importances(self, node, weight):
        """Recursively calculate feature importances."""
        if node.feature is not None:
            self.feature_importances_[node.feature] += weight
            left_weight = weight * 0.5  # Simple weighting scheme
            right_weight = weight * 0.5
            self._calculate_feature_importances(node.left, left_weight)
            self._calculate_feature_importances(node.right, right_weight)
    
    def predict_sample(self, x, node):
        """Predict class for a single sample."""
        if node.value is not None:
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self.predict_sample(x, node.left)
        return self.predict_sample(x, node.right)
    
    def predict(self, X):
        """
        Predict classes for multiple samples.
        
        Parameters:
        X (array-like): Samples to predict
        
        Returns:
        array: Predicted classes
        """
        return np.array([self.predict_sample(x, self.root) for x in X])

def generate_dataset(n_samples=1000, n_features=5, n_classes=3):
    """
    Generate synthetic dataset suitable for decision tree classification.
    
    Parameters:
    n_samples (int): Number of samples to generate
    n_features (int): Number of features
    n_classes (int): Number of classes
    
    Returns:
    tuple: (features array, labels array)
    """
    np.random.seed(42)
    
    X = np.random.randn(n_samples, n_features)
    
    # Create non-linear decision boundaries
    y = np.zeros(n_samples, dtype=int)
    
    # Define complex rules for classification
    for i in range(n_samples):
        if X[i, 0] > 0.5 and X[i, 1] > 0:
            y[i] = 0
        elif X[i, 0] < -0.5 and X[i, 2] > 0:
            y[i] = 1
        elif np.abs(X[i, 3]) > 1 and X[i, 4] < 0:
            y[i] = 2
        else:
            y[i] = np.random.randint(0, n_classes)
    
    return X, y

# Generate dataset
print("Generating synthetic dataset...")
X, y = generate_dataset()

# Split data into train and test sets
def train_test_split(X, y, test_size=0.2):
    """Split data into training and testing sets."""
    n_test = int(len(X) * test_size)
    indices = np.random.permutation(len(X))
    test_idx, train_idx = indices[:n_test], indices[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

X_train, X_test, y_train, y_test = train_test_split(X, y)

# Train and evaluate decision tree
print("\nTraining decision tree...")
dt = DecisionTree(max_depth=10, min_samples_split=5)
dt.fit(X_train, y_train)

# Make predictions
print("Making predictions...")
y_pred = dt.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot feature importances
def plot_feature_importances(importances):
    """Plot feature importances."""
    plt.figure(figsize=(10, 6))
    features = [f'Feature {i+1}' for i in range(len(importances))]
    plt.bar(features, importances)
    plt.title('Feature Importances')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

print("\nPlotting feature importances...")
plot_feature_importances(dt.feature_importances_)

# Function to visualize tree structure (for small trees)
def visualize_tree(node, depth=0, feature_names=None):
    """Visualize tree structure."""
    if node is None:
        return
    
    indent = "  " * depth
    if node.value is not None:
        print(f"{indent}Predict class: {node.value}")
    else:
        feature_name = f"Feature {node.feature}" if feature_names is None else feature_names[node.feature]
        print(f"{indent}{feature_name} <= {node.threshold:.2f}")
        visualize_tree(node.left, depth + 1, feature_names)
        visualize_tree(node.right, depth + 1, feature_names)

# Visualize first few levels of the tree
print("\nTree Structure (first few levels):")
visualize_tree(dt.root, depth=0, feature_names=None)