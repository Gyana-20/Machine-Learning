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


# Step 2: Implement the LDA Algorithm from Scratch

class LDA:

    def fit(self, X, y):

        n_samples, n_features = X.shape

        self.classes = np.unique(y)

        n_classes = len(self.classes)


        # Compute the mean vectors for each class

        self.means = np.array([X[y == cls].mean(axis=0) for cls in self.classes])


        # Compute the within-class scatter matrix

        self.Sw = np.zeros((n_features, n_features))

        for cls in self.classes:

            class_scatter = np.cov(X[y == cls].T)

            self.Sw += class_scatter


        # Compute the overall mean

        overall_mean = X.mean(axis=0)


        # Compute the between-class scatter matrix

        self.Sb = np.zeros((n_features, n_features))

        for cls, mean in zip(self.classes, self.means):

            n_cls = X[y == cls].shape[0]

            mean_diff = (mean - overall_mean).reshape(n_features, 1)

            self.Sb += n_cls * (mean_diff).dot(mean_diff.T)


        # Compute the eigenvalues and eigenvectors

        eigvals, eigvecs = np.linalg.eig(np.linalg.inv(self.Sw).dot(self.Sb))


        # Sort the eigenvectors by eigenvalues in descending order

        sorted_indices = np.argsort(eigvals)[::-1]

        self.eigenvectors = eigvecs[:, sorted_indices]

        self.eigenvalues = eigvals[sorted_indices]


        # Select the top eigenvector (1D projection)

        self.W = self.eigenvectors[:, :1]


    def transform(self, X):

        return X.dot(self.W)


# Step 3: Evaluate the Model

def evaluate_model(X, y):

    # Split the dataset into training and testing sets

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    

    # Train the LDA

    lda = LDA()

    lda.fit(X_train, y_train)

    

    # Transform the test data

    X_test_lda = lda.transform(X_test)

    

    # Make predictions based on the transformed data

    predictions = np.where(X_test_lda > 0, 1, 0)

    

    # Calculate accuracy

    accuracy = accuracy_score(y_test, predictions)

    return accuracy


# Create a synthetic dataset

X, y = generate_data(n_samples=1000)


# Evaluate the model

accuracy = evaluate_model(X, y)

print(f"Accuracy of the LDA: {accuracy:.4f}")


# Visualize the results

def visualize_results(X, y, lda):

    X_lda = lda.transform(X)

    

    plt.figure(figsize=(10, 6))

    plt.scatter(X_lda[y == 0], np.zeros((X_lda[y == 0].shape[0])), color='red', label='Class 0', alpha=0.5 )

    plt.scatter(X_lda[y == 1], np.zeros((X_lda[y == 1].shape[0])), color='blue', label='Class 1', alpha=0.5)

    plt.title("LDA Projection")

    plt.xlabel("LDA Component")

    plt.yticks([])

    plt.legend()

    plt.show()


# Visualize the results

lda = LDA()

lda.fit(X, y)

visualize_results(X, y, lda)