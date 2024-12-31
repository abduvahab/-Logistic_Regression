import pandas as pd
import numpy as np
import argparse
import os
import math


def check_file(f_name: str):
    if os.path.exists(f_name):
        if f_name.endswith('.csv') is False:
            raise Exception("the file should be a .csv file")
    else:
        raise Exception("the file doesn't exist")


def check_content_of_file(f_name: str) -> pd.DataFrame:
    df = pd.read_csv(f_name)
    if df is None:
        raise Exception("DataFrame is Nne")
    elif df.empty:
        raise Exception("DataFrame  is empty")
    else:
        return df


def read_dataset() -> pd.DataFrame:
    parser = argparse.ArgumentParser(
        prog='describe.py',
        usage='%(prog)s file_name.csv',
        description='the description for the dataset about numeric features'
    )
    parser.add_argument('f_name', help='the name of the datset file')
    args = parser.parse_args()
    f_name = args.f_name
    # check if the file exist
    check_file(f_name)
    # check if the file  is empty
    df = check_content_of_file(f_name)
    return df


def find_std(col: pd.Series, mean, num):
    ndarr = col.values.astype(float)
    ndarr -= mean
    ndarr = np.square(ndarr)
    sum_ndarr = 0
    for x in np.nditer(ndarr):
        sum_ndarr += x
    std = math.sqrt(sum_ndarr / num)
    return std


def find_sum(col: pd.Series):
    ndarr = col.values
    sums = 0
    for x in np.nditer(ndarr):
        sums += x
    return sums


def find_min(col: pd.Series):
    mins = col[0]
    for x in np.nditer(col):
        if mins > x:
            mins = x
    return mins


def find_max(col: pd.Series):
    maxs = col[0]
    for x in np.nditer(col):
        if maxs < x:
            maxs = x
    return maxs


def order_array(col: pd.Series) -> np.ndarray:
    ndarr = col.values.copy()
    num = len(ndarr)
    for i in range(0, num-1):
        for j in range(i+1, num):
            if ndarr[i] > ndarr[j]:
                tem = ndarr[i]
                ndarr[i] = ndarr[j]
                ndarr[j] = tem
    # for i in range(0, num-1):
    #     for j in range(0, num -i -1):
    #         if ndarr[j] > ndarr[j+1]:
    #             tem = ndarr[j]
    #             ndarr[j] = ndarr[j+1]
    #             ndarr[j+1] = tem
    return ndarr


def find_percentage(ndarr: np.ndarray, per):
    num = len(ndarr)
    R = (per / 100) * (num - 1)
    l_index = int(R)
    u_index = l_index + 1
    weight = R - l_index
    if u_index < num:
        t = ndarr[l_index] + weight * (ndarr[u_index] - ndarr[l_index])
        return t
    else:
        return ndarr[l_index]


def calculate_des_value(des: pd.DataFrame, col: pd.Series, name_col: str):
    des[name_col] = np.nan
    count = col.size
    des.loc["count", name_col] = count
    total = find_sum(col)
    mean = total / count
    des.loc["mean", name_col] = mean
    std = find_std(col, mean, count)
    des.loc["std", name_col] = std
    mins = find_min(col)
    des.loc["min", name_col] = mins
    maxs = find_max(col)
    des.loc["max", name_col] = maxs
    ordered_arr = order_array(col)
    f_25 = find_percentage(ordered_arr, 25)
    des.loc["25%", name_col] = f_25
    f_50 = find_percentage(ordered_arr, 50)
    des.loc["50%", name_col] = f_50
    f_75 = find_percentage(ordered_arr, 75)
    des.loc["75%", name_col] = f_75


def get_all_describe(des: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].isnull().all():
            des[col] = np.nan
            des.loc['count', col] = 0
            # print(des)
        elif np.issubdtype(df[col].dtype, np.number):
            n_col = df[col].dropna().copy()
            calculate_des_value(des, n_col, col)

        # print(df[col].dtype)
        # print(type(df[col]))
    pass


def main():
    try:
        # read file and check the content
        df = read_dataset()
        # create a dataFame with index {"count", "mean", ...}
        index = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
        des = pd.DataFrame(index=index)
        # get all the value from dataset
        get_all_describe(des, df)
        print(des)
        print(df.describe())

    except Exception as e:
        print(f"Error: {e}")
        pass


if __name__ == "__main__":
    main()
