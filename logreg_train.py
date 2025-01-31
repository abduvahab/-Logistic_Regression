import pandas as pd 
import numpy as np


def read_dataset() -> pd.DataFrame:
    f_name = "./datasets/dataset_train.csv"
    df = pd.read_csv(f_name)
    # print(df.info())
    return df


def clean_datset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna()
    df.reset_index(drop=True, inplace=True)
    # print(df.info())
    return df

def codeing_houses(y: pd.Series):
    code_house = {}
    uniques = y.unique()
    for i in range(len(uniques)):
        code_house[uniques[i]] = i
    return y.map(code_house), code_house


def sigmoid_function(Z):
    return (1 /(1 + np.exp(-Z)))


def loss_function(y, y_predict):
    y_predict = np.clip(y_predict, 1e-15, 1 - 1e-15)
    return -np.mean(y * np.log(y_predict) + (1 - y) * np.log(1 - y_predict))


def gradiant_descent(X, y, weights, l_rate, epoch):
    for i in range(epoch):
        y_predict = sigmoid_function(np.dot(X, weights.T))
        gradiant = np.dot(X.T, (y - y_predict)) / y.size
        weights += l_rate * gradiant
        if i % 100 == 0:
            loss = loss_function(y, y_predict)
            # print(f"iteartion {i}, loss: {loss}")
        # break
    return weights



def create_model(df: pd.DataFrame, l_rate=0.01, epoch=5000) -> pd.DataFrame:
    #create the features
    X = df[['Astronomy','Herbology']].values
    #create target
    y = df['Hogwarts House']
    #encoding the target
    y_code, code_house = codeing_houses(y)
    #normalization the target
    X_std = (X - X.mean(axis=0))/ X.std(axis=0)
    y_code = y_code.values
    #add bias to features
    x_std = np.c_[np.ones(X_std.shape[0]), X_std]
    weights = np.random.rand(len(code_house), x_std.shape[1]) * 0.01
    # print(weights)
    for i, c in enumerate(code_house):
        y_binary = (y_code == code_house[c]).astype(int)
        weights[i] = gradiant_descent(x_std, y_binary, weights[i], l_rate, epoch)
        # break
    # print(weights)
    return weights, code_house


def get_precision(y_true, y_pred, num_classes):
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


def estimate_result(df: pd.DataFrame, weights, code_house):
    X = df[['Astronomy','Herbology']].values
    x_std = (X - X.mean(axis=0)) / X.std(axis=0)
    x_std = np.c_[np.ones(X.shape[0]), x_std]
    y_predict  = sigmoid_function(np.dot(x_std, weights.T))
    print(y_predict)
    y_predict = np.argmax(y_predict, axis=1)
    print(code_house)
    y_true = df['Hogwarts House'].map(code_house)
    print(y_predict)
    print(y_true.values)
    get_precision(y_true.values, y_predict, len(code_house))


def main():
    try:
        #read the dataset
        df = read_dataset()
        #clean dataset
        df = clean_datset(df)
        #create model
        weights, code_house = create_model(df)
        estimate_result(df,weights, code_house)

    except Exception as e:
        print(f"Error:{e}")
    pass


if __name__ == "__main__":
    main()