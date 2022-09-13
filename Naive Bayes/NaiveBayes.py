import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

class NaiveBayes:
    def __init__(self):
        self.classes = None
        self.mean = None
        self.var = None
        self.priors = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        # Initialize parameters
        self.mean = np.zeros((n_classes, n_features))
        self.var = np.zeros((n_classes, n_features))
        self.priors = np.zeros(n_classes)
        
        # Calculate mean, variance, and prior for each class
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0)
            self.priors[idx] = X_c.shape[0] / float(n_samples)
    
    def gaussian_density(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(-(x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
    
    def predict(self, X):
        y_pred = []
        
        for x in X:
            posteriors = []
            
            # Calculate posterior probability for each class
            for idx, c in enumerate(self.classes):
                prior = np.log(self.priors[idx])
                posterior = np.sum(np.log(self.gaussian_density(idx, x)))
                posterior = prior + posterior
                posteriors.append(posterior)
            
            # Select class with highest posterior probability
            y_pred.append(self.classes[np.argmax(posteriors)])
        
        return np.array(y_pred)

# Generate synthetic dataset
def generate_dataset(n_samples=1000):
    np.random.seed(42)
    
    # Generate features for class 0
    n_class_0 = n_samples // 2
    feature1_class0 = np.random.normal(0, 1, n_class_0)
    feature2_class0 = np.random.normal(2, 1, n_class_0)
    feature3_class0 = np.random.normal(-1, 0.5, n_class_0)
    
    # Generate features for class 1
    n_class_1 = n_samples - n_class_0
    feature1_class1 = np.random.normal(3, 1, n_class_1)
    feature2_class1 = np.random.normal(0, 1, n_class_1)
    feature3_class1 = np.random.normal(1, 0.5, n_class_1)
    
    # Combine features
    X = np.vstack([
        np.column_stack((feature1_class0, feature2_class0, feature3_class0)),
        np.column_stack((feature1_class1, feature2_class1, feature3_class1))
    ])
    
    # Create labels
    y = np.hstack([np.zeros(n_class_0), np.ones(n_class_1)])
    
    return X, y

# Generate and split the dataset
X, y = generate_dataset(1000)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate the model
nb = NaiveBayes()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)

# Calculate and display metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualize the first two features
plt.figure(figsize=(10, 6))
plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], 
           alpha=0.5, label='Class 0', color='blue')
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], 
           alpha=0.5, label='Class 1', color='red')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Synthetic Dataset Visualization')
plt.legend()
plt.grid(True)
plt.show()
