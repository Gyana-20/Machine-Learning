import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Generate synthetic dataset
np.random.seed(42)
N = 100000  
X = 10 * np.random.rand(N, 1)  
y = 5 * X + 3 + np.random.randn(N, 1) * 5  

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.lr = learning_rate
        self.epochs = epochs
        self.theta = None
        self.loss_history = []

    def compute_loss(self, X, y):
        m = len(y)
        return (1 / (2 * m)) * np.sum((X.dot(self.theta) - y) ** 2)

    def fit(self, X, y):
        m, n = X.shape
        X_b = np.c_[np.ones((m, 1)), X]  
        self.theta = np.random.randn(n + 1, 1)  

        for epoch in range(self.epochs):
            gradients = (1 / m) * X_b.T.dot(X_b.dot(self.theta) - y)
            self.theta -= self.lr * gradients
            self.loss_history.append(self.compute_loss(X_b, y))
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {self.loss_history[-1]:.4f}")

    def predict(self, X):
        return np.c_[np.ones((X.shape[0], 1)), X].dot(self.theta)

# Train the model
model = LinearRegression(learning_rate=0.01, epochs=100)

model.fit(X_train, y_train)

# Evaluate on test data
y_pred = model.predict(X_test)
test_loss = np.mean((y_pred - y_test) ** 2) / 2
print(f"Final Test Loss: {test_loss:.4f}")

# Plot Loss Function
plt.plot(model.loss_history, label="Training Loss", color='blue')
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.title("Loss Function Over Time")
plt.legend()
plt.grid()
plt.show()

# Plot Data & Regression Line
plt.scatter(X_test[:1000], y_test[:1000], alpha=0.2, color="gray", label="Test Data")
plt.plot(X_test, y_pred, color='red', linewidth=2, label="Regression Line")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression Fit")
plt.legend()
plt.grid()
plt.show()
