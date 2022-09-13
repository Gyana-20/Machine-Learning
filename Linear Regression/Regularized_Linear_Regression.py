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

class RegularizedLinearRegression:
    def __init__(self, learning_rate=0.01, epochs=100, alpha=0.1, regularization="ridge"):
        self.lr = learning_rate
        self.epochs = epochs
        self.alpha = alpha
        self.regularization = regularization
        self.theta = None
        self.loss_history = []

    def compute_loss(self, X, y):
        m = len(y)
        y_pred = X.dot(self.theta)
        mse_loss = (1 / (2 * m)) * np.sum((y_pred - y) ** 2)
        
        if self.regularization == "ridge":
            reg_loss = (self.alpha / (2 * m)) * np.sum(self.theta[1:] ** 2)
        elif self.regularization == "lasso":
            reg_loss = (self.alpha / (2 * m)) * np.sum(np.abs(self.theta[1:]))
        else:
            reg_loss = 0
        
        return mse_loss + reg_loss

    def fit(self, X, y):
        m, n = X.shape
        X_b = np.c_[np.ones((m, 1)), X]  
        self.theta = np.random.randn(n + 1, 1)  

        for epoch in range(self.epochs):
            gradients = (1 / m) * X_b.T.dot(X_b.dot(self.theta) - y)

            if self.regularization == "ridge":
                gradients[1:] += (self.alpha / m) * self.theta[1:]  
            elif self.regularization == "lasso":
                gradients[1:] += (self.alpha / m) * np.sign(self.theta[1:])  

            self.theta -= self.lr * gradients
            self.loss_history.append(self.compute_loss(X_b, y))

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {self.loss_history[-1]:.4f}")

    def predict(self, X):
        return np.c_[np.ones((X.shape[0], 1)), X].dot(self.theta)

# Train Ridge Regression
ridge_model = RegularizedLinearRegression(learning_rate=0.01, epochs=100, alpha=0.1, regularization="ridge")
ridge_model.fit(X_train, y_train)

# Train Lasso Regression
lasso_model = RegularizedLinearRegression(learning_rate=0.01, epochs=100, alpha=0.1, regularization="lasso")
lasso_model.fit(X_train, y_train)

# Evaluate
ridge_pred = ridge_model.predict(X_test)
lasso_pred = lasso_model.predict(X_test)

ridge_loss = np.mean((ridge_pred - y_test) ** 2) / 2
lasso_loss = np.mean((lasso_pred - y_test) ** 2) / 2

print(f"Ridge Test Loss: {ridge_loss:.4f}")
print(f"Lasso Test Loss: {lasso_loss:.4f}")

# Plot Loss Function
plt.plot(ridge_model.loss_history, label="Ridge Loss", color='blue')
plt.plot(lasso_model.loss_history, label="Lasso Loss", color='green')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Function Over Time")
plt.legend()
plt.grid()
plt.show()

# Plot Data & Regression Line
plt.scatter(X_test[:1000], y_test[:1000], alpha=0.2, color="gray", label="Test Data")
plt.plot(X_test, ridge_pred, color='red', linewidth=2, label="Ridge Regression")
plt.plot(X_test, lasso_pred, color='blue', linewidth=2, label="Lasso Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Ridge vs Lasso Regression Fit")
plt.legend()
plt.grid()
plt.show()
