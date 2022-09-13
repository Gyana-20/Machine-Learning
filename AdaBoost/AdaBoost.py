import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Step 1: Create a synthetic dataset
def create_synthetic_data(n_samples=1000):
    np.random.seed(42)  # For reproducibility
    # Generate random points for class 0
    X0 = np.random.randn(n_samples // 2, 2) + np.array([-2, -2])
    # Generate random points for class 1
    X1 = np.random.randn(n_samples // 2, 2) + np.array([2, 2])
    # Combine the data
    X = np.vstack((X0, X1))
    # Create labels
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))
    return X, y

# Step 2: Implement the AdaBoost Algorithm from Scratch
class AdaBoost:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.alphas = []
        self.models = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        # Initialize weights
        w = np.ones(n_samples) / n_samples

        for _ in range(self.n_estimators):
            # Create a weak classifier (decision stump)
            model = DecisionTreeClassifier(max_depth=1)
            model.fit(X, y, sample_weight=w)
            y_pred = model.predict(X)

            # Calculate the error
            error = np.sum(w * (y_pred != y)) / np.sum(w)

            # Calculate the alpha value
            alpha = 0.5 * np.log((1 - error) / (error + 1e-10))  # Avoid division by zero
            self.alphas.append(alpha)
            self.models.append(model)

            # Update weights
            w *= np.exp(-alpha * (2 * y_pred - 1))  # y_pred is 0 or 1, so we convert it to -1 or 1
            w /= np.sum(w)  # Normalize weights

    def predict(self, X):
        # Aggregate predictions from all weak classifiers
        final_pred = np.zeros(X.shape[0])
        for alpha, model in zip(self.alphas, self.models):
            final_pred += alpha * model.predict(X)
        return np.where(final_pred > 0, 1, 0)

# Step 3: Evaluate the Model
def evaluate_model(X, y):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the AdaBoost model
    ada = AdaBoost(n_estimators=50)
    ada.fit(X_train, y_train)
    
    # Make predictions
    predictions = ada.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    return accuracy

# Create a synthetic dataset
X, y = create_synthetic_data(n_samples=1000)

# Evaluate the model
accuracy = evaluate_model(X, y)
print(f"Accuracy of the AdaBoost: {accuracy:.4f}")

# Visualize the decision boundary
def plot_decision_boundary(X, y, model):
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k', s=50)
    
    # Create grid to plot decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.title("AdaBoost Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

# Visualize the decision boundary
ada = AdaBoost(n_estimators=50)
ada.fit(X, y)
plot_decision_boundary(X, y, ada)