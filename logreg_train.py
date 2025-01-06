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

def main():
    try:
        #read the dataset
        df = read_dataset()
        #clean dataset
        df = clean_datset(df)

    except Exception as e:
        print(f"Error:{e}")
    pass


if __name__ == "__main__":
    main()