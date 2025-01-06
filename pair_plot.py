import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    f_name = "./datasets/dataset_train.csv"
    df = pd.read_csv(f_name)
    df = df.dropna()
    df.reset_index(drop=True, inplace=True)
    ft_col = ['Index', 'First Name', 'Last Name', 'Birthday', 'Best Hand']
    n_df = df.drop(columns=ft_col)
    sns.pairplot(n_df, hue='Hogwarts House', diag_kind='kde')
    plt.show()


if __name__ == "__main__":
    main()
