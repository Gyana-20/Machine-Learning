import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []
        
    def sigmoid(self, z):
        """Compute the sigmoid function."""
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))
    
    def fit(self, X, y):
        """Train the logistic regression model."""
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Forward pass
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)
            
            # Calculate loss
            loss = -np.mean(y * np.log(y_predicted + 1e-15) + 
                          (1 - y) * np.log(1 - y_predicted + 1e-15))
            self.loss_history.append(loss)
            
            # Calculate gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Print loss every 100 iterations
            if (i + 1) % 100 == 0:
                print(f'Iteration {i+1}/{self.n_iterations}, Loss: {loss:.4f}')
    
    def predict(self, X):
        """Make predictions for given input data."""
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        return (y_predicted > 0.5).astype(int)

# Generate synthetic dataset
def generate_dataset(n_samples=100000, n_features=2):
    """Generate synthetic dataset for binary classification."""
    np.random.seed(42)
    
    # Generate feature matrix
    X = np.random.randn(n_samples, n_features)
    
    # Generate target variable
    # Create a non-linear decision boundary
    boundary = X[:, 0]**2 + X[:, 1]**2
    y = (boundary < 4).astype(int)
    
    # Add some noise
    noise = np.random.random(n_samples) < 0.05
    y[noise] = 1 - y[noise]
    
    return X, y

# Generate the dataset
X, y = generate_dataset()

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Performance:")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot the loss history
plt.figure(figsize=(10, 6))
plt.plot(model.loss_history)
plt.title('Loss History')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

# Visualize decision boundary (for 2D data)
plt.figure(figsize=(10, 8))
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
plt.title('Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()