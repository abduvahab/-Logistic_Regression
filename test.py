import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

def read_data():
    # Load the Iris dataset
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['Species'] = iris.target

    # Map target values to species names
    # species_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
    # df['Species'] = df['Species'].map(species_map)

    # print(df.info())
    # print(df)

    # Create a pair plot
    # sns.pairplot(df, hue="Species", diag_kind="kde")
    # plt.show()
    return df


def sigmoid(X):
    return 1 / (1 + np.exp(-X))


def loss_computation(y, y_predict):
    return -np.mean(y * np.log(y_predict) + (1 - y) * np.log(1 - y_predict))

def gradiant_descent(X, y, weights, l_rate, epoch):

    for i in range(epoch):
        y_predict = sigmoid(np.dot(X, weights.T))
        gradient = np.dot(X.T, (y - y_predict))/(y.size)
        weights += l_rate * gradient
        if i % 100 == 0 :
            loss = loss_computation(y, y_predict)
            # print(f"itaretion {i} , loss:{loss:4f}")
    # print(f"the weights:\n{weights}")
    return weights
    pass


def train_model(df: pd.DataFrame, l_rate=0.005, epoch=10000):

    #get features and target columns
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    #get classes
    # classes = df['Species'].unique()
    # Get unique classes
    classes = np.unique(y)
    # Normalize features
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    #add bias to features
    X = np.c_[np.ones(X.shape[0]), X]

    # weights
    num_classes = len(classes)
    # weights = np.zeros((num_classes, X.shape[1]))
    weights = np.random.rand(num_classes, X.shape[1]) * 0.01  # Small random weights

    for i, c in enumerate(classes):
        #(one vs rest) in the target , right value is 1, the rest wrong vlues is 0
        y_binary = (y==c).astype(int)
        #gradiant descent
        weights[i] = gradiant_descent(X, y_binary, weights[i], l_rate, epoch)
        
    print(weights)
    return weights
    # print(X)
    # print(y)
    pass

def predict_multiclass(X, weights):
    # Normalize features (same as during training)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    X = np.c_[np.ones(X.shape[0]), X]
    y_predict  =sigmoid(np.dot(X, weights.T))
    # print(y_predict)
    y_predict = np.argmax(y_predict, axis=1)
    print(y_predict)
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
    y_predict = predict_multiclass(df.iloc[:, :-1], weights)
    y_true = df['Species'].values
    num_classes = len(np.unique(y_true))
    
    # Calculate precision
    calculate_precision(y_true, y_predict, num_classes)
    pass


if __name__ == "__main__":
    main()