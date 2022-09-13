import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score


# Step 1: Create a synthetic dataset

def generate_data(n_samples=1000):

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


# Step 2: Implement the Perceptron Algorithm from Scratch

class Perceptron:

    def __init__(self, learning_rate=0.01, n_iterations=1000):

        self.learning_rate = learning_rate

        self.n_iterations = n_iterations

        self.weights = None

        self.bias = None


    def fit(self, X, y):

        n_samples, n_features = X.shape

        # Initialize weights and bias

        self.weights = np.zeros(n_features)

        self.bias = 0


        # Training the Perceptron

        for _ in range(self.n_iterations):

            for idx, x_i in enumerate(X):

                linear_output = np.dot(x_i, self.weights) + self.bias

                y_predicted = self._activation_function(linear_output)


                # Update weights and bias if misclassified

                if y_predicted != y[idx]:

                    update = self.learning_rate * (y[idx] - y_predicted)

                    self.weights += update * x_i

                    self.bias += update


    def predict(self, X):

        linear_output = np.dot(X, self.weights) + self.bias

        y_predicted = self._activation_function(linear_output)

        return y_predicted


    def _activation_function(self, x):

        return np.where(x >= 0, 1, 0)


# Step 3: Evaluate the Model

def evaluate_model(X, y):

    # Split the dataset into training and testing sets

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    

    # Train the Perceptron

    perceptron = Perceptron(learning_rate=0.01, n_iterations=1000)

    perceptron.fit(X_train, y_train)

    

    # Make predictions

    predictions = perceptron.predict(X_test)

    

    # Calculate accuracy

    accuracy = accuracy_score(y_test, predictions)

    return accuracy


# Create a synthetic dataset

X, y = generate_data(n_samples=1000)


# Evaluate the model

accuracy = evaluate_model(X, y)

print(f"Accuracy of the Perceptron: {accuracy:.4f}")


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

    plt.title ("Perceptron Decision Boundary")

    plt.xlabel("Feature 1")

    plt.ylabel("Feature 2")

    plt.show()


# Plot the decision boundary

perceptron = Perceptron(learning_rate=0.01, n_iterations=1000)

perceptron.fit(X, y)

plot_decision_boundary(X, y, perceptron)