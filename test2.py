import numpy as np
import pandas as pd
from sklearn.datasets import load_iris


def read_data():
    # Load the Iris dataset
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['Species'] = iris.target
    return df


def sigmoid(X):
    return 1 / (1 + np.exp(-X))


def loss_computation(y, y_predict):
    # Avoid log(0) by clipping predictions
    y_predict = np.clip(y_predict, 1e-15, 1 - 1e-15)
    return -np.mean(y * np.log(y_predict) + (1 - y) * np.log(1 - y_predict))


def gradient_descent(X, y, weights, l_rate, epoch):
    for i in range(epoch):
        y_predict = sigmoid(np.dot(X, weights.T))
        gradient = np.dot(X.T, (y - y_predict)) / y.size
        weights += l_rate * gradient  # Update weights
        if i % 100 == 0:  # Print loss every 100 iterations
            loss = loss_computation(y, y_predict)
            # print(f"Iteration {i}, Loss: {loss:.6f}")
    return weights


def train_model(df: pd.DataFrame, l_rate=0.01, epoch=5000):
    # Get features and target columns
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Normalize features
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    # Add bias to features
    X = np.c_[np.ones(X.shape[0]), X]

    # Get unique classes
    classes = np.unique(y)

    # Initialize weights
    num_classes = len(classes)
    weights = np.random.rand(num_classes, X.shape[1]) * 0.01  # Small random weights

    for i, c in enumerate(classes):
        # One-vs-rest for the target
        y_binary = (y == c).astype(int)
        # Gradient descent
        weights[i] = gradient_descent(X, y_binary, weights[i], l_rate, epoch)

    return weights


def predict_multiclass(X, weights):
    # Normalize features (same as during training)
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    # Add bias to features
    X = np.c_[np.ones(X.shape[0]), X]

    # Predict probabilities for each class
    y_predict = sigmoid(np.dot(X, weights.T))

    # Choose the class with the highest probability
    y_predict = np.argmax(y_predict, axis=1)
    return y_predict

def calculate_precision(y_true, y_pred, num_classes):
    precision = np.zeros(num_classes)

    for c in range(num_classes):
        # True Positives (TP) for class c
        TP = np.sum((y_pred == c) & (y_true == c))
        # False Positives (FP) for class c
        FP = np.sum((y_pred == c) & (y_true != c))
        # Precision for class c
        if TP + FP > 0:
            precision[c] = TP / (TP + FP)
        else:
            precision[c] = 0  # To avoid division by zero

    macro_precision = np.mean(precision)
    
    print("Per-Class Precision:", precision)
    print("Macro-Average Precision:", macro_precision)
    return precision, macro_precision

def main():
    df = read_data()
    weights = train_model(df)
    y_predict = predict_multiclass(df.iloc[:, :-1].values, weights)
    y_true = df['Species'].values
    num_classes = len(np.unique(y_true))
    
    # Calculate precision
    calculate_precision(y_true, y_predict, num_classes)
    # print("Predicted classes for all items:")
    # print(y_predict)
    # print("Actual classes:")
    # print(df['Species'].values)


if __name__ == "__main__":
    main()
