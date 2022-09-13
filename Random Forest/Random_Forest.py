import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm

class DecisionTreeNode:
    """Node class for Decision Trees within Random Forest."""
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree:
    """Decision Tree Classifier for Random Forest."""
    def __init__(self, max_depth=10, min_samples_split=2, n_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.root = None
    
    def _entropy(self, y):
        """Calculate entropy of label distribution."""
        proportions = np.bincount(y) / len(y)
        return -np.sum([p * np.log2(p) for p in proportions if p > 0])
    
    def _information_gain(self, y, y_left, y_right):
        """Calculate information gain for a split."""
        p_left = len(y_left) / len(y)
        p_right = len(y_right) / len(y)
        return self._entropy(y) - (p_left * self._entropy(y_left) + p_right * self._entropy(y_right))
    
    def _best_split(self, X, y, feature_indices):
        """Find best split considering only a subset of features."""
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        for feature in feature_indices:
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) < self.min_samples_split or np.sum(right_mask) < self.min_samples_split:
                    continue
                
                gain = self._information_gain(y, y[left_mask], y[right_mask])
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        
        # Stopping criteria
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            len(np.unique(y)) == 1):
            return DecisionTreeNode(value=np.argmax(np.bincount(y)))
        
        # Randomly select subset of features
        feature_indices = np.random.choice(n_features, 
                                         size=self.n_features, 
                                         replace=False)
        
        # Find best split
        best_feature, best_threshold = self._best_split(X, y, feature_indices)
        
        if best_feature is None:
            return DecisionTreeNode(value=np.argmax(np.bincount(y)))
        
        # Split data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        # Create child nodes
        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return DecisionTreeNode(best_feature, best_threshold, left, right)
    
    def fit(self, X, y):
        """Train the decision tree."""
        self.n_features = self.n_features or max(1, int(np.sqrt(X.shape[1])))
        self.root = self._build_tree(X, y)
    
    def _predict_sample(self, x, node):
        """Predict class for a single sample."""
        if node.value is not None:
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._predict_sample(x, node.left)
        return self._predict_sample(x, node.right)
    
    def predict(self, X):
        """Predict classes for multiple samples."""
        return np.array([self._predict_sample(x, self.root) for x in X])

class RandomForest:
    """Random Forest Classifier implementation."""
    def __init__(self, n_trees=100, max_depth=10, min_samples_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []
        
    def fit(self, X, y):
        """Train the random forest."""
        self.trees = []
        n_samples = X.shape[0]
        
        for _ in tqdm(range(self.n_trees), desc="Training trees"):
            # Bootstrap sampling
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            sample_X = X[indices]
            sample_y = y[indices]
            
            # Create and train tree
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=self.n_features
            )
            tree.fit(sample_X, sample_y)
            self.trees.append(tree)
    
    def predict(self, X):
        """Predict classes using majority voting."""
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.array([np.argmax(np.bincount(predictions[:, i])) 
                        for i in range(len(X))])

def generate_dataset(n_samples=1000, n_features=10, n_classes=3):
    """Generate synthetic dataset for random forest classification."""
    np.random.seed(42)
    
    X = np.random.randn(n_samples, n_features)
    y = np.zeros(n_samples, dtype=int)
    
    # Create complex decision rules
    for i in range(n_samples):
        # Rule 1: Combination of first three features
        if X[i, 0] > 0 and X[i, 1] > 0 and X[i, 2] > 0:
            y[i] = 0
        # Rule 2: Interaction between features 3-5
        elif X[i, 3] * X[i, 4] > 0 and X[i, 5] < 0:
            y[i] = 1
        # Rule 3: Non-linear combination of features 6-8
        elif np.sum(X[i, 6:9]**2) > 3:
            y[i] = 2
        else:
            y[i] = np.random.randint(0, n_classes)
            
    return X, y

# Generate dataset
print("Generating synthetic dataset...")
X, y = generate_dataset()

# Split data
def train_test_split(X, y, test_size=0.2):
    """Split data into training and testing sets."""
    n_test = int(len(X) * test_size)
    indices = np.random.permutation(len(X))
    test_idx, train_idx = indices[:n_test], indices[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

X_train, X_test, y_train, y_test = train_test_split(X, y)

# Train random forest
print("\nTraining Random Forest...")
rf = RandomForest(n_trees=100, max_depth=10, min_samples_split=2)
rf.fit(X_train, y_train)

# Make predictions
print("Making predictions...")
y_pred = rf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Analyze feature importance through permutation
def calculate_feature_importance(model, X, y, n_repeats=5):
    """Calculate feature importance using permutation importance."""
    base_score = accuracy_score(y, model.predict(X))
    importances = np.zeros(X.shape[1])
    
    for i in range(X.shape[1]):
        scores = np.zeros(n_repeats)
        for j in range(n_repeats):
            X_permuted = X.copy()
            X_permuted[:, i] = np.random.permutation(X[:, i])
            scores[j] = base_score - accuracy_score(y, model.predict(X_permuted))
        importances[i] = np.mean(scores)
    
    return importances

# Calculate and plot feature importances
print("\nCalculating feature importances...")
importances = calculate_feature_importance(rf, X_test, y_test)

plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances)
plt.title('Feature Importances in Random Forest')
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.tight_layout()
plt.show()

# Plot learning curves
def plot_learning_curves(X, y, model, n_points=5):
    """Plot learning curves for the random forest."""
    train_sizes = np.linspace(0.1, 1.0, n_points)
    train_scores = []
    test_scores = []
    
    for size in train_sizes:
        n_train = int(len(X) * size)
        indices = np.random.choice(len(X), size=n_train, replace=False)
        X_subset = X[indices]
        y_subset = y[indices]
        
        X_train, X_test, y_train, y_test = train_test_split(X_subset, y_subset)
        model.fit(X_train, y_train)
        
        train_scores.append(accuracy_score(y_train, model.predict(X_train)))
        test_scores.append(accuracy_score(y_test, model.predict(X_test)))

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores, label='Training Score')
    plt.plot(train_sizes, test_scores, label='Testing Score')
    plt.title('Learning Curves')
    plt.xlabel('Training Set Size (%)')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

print("\nPlotting learning curves...")
plot_learning_curves(X, y, RandomForest(n_trees=50, max_depth=8))