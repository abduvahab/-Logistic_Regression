import pandas as pd
import numpy as np 
import argparse
import os

def check_file(f_name:str):
    if os.path.exists(f_name):
        if f_name.endswith('.csv') is False:
            raise Exception("the file should be a .csv file")
    else:
        raise Exception("the file don't exist")


def check_content_of_file(f_name:str)->pd.DataFrame:
    df = pd.read_csv(f_name)
    print(df.info())
    print(df.describe())
    return df


def read_dataset()->pd.DataFrame:
    parser = argparse.ArgumentParser(
        prog = 'describe.py',
        usage = '%(prog)s file_name.csv',
        description = 'give the description for the dataset about numeric features'
    )
    parser.add_argument('f_name', help='the name of the datset file')
    args = parser.parse_args()
    print(type(args))
    f_name = args.f_name
    # check if the file exist or not 
    check_file(f_name)
    check_content_of_file(f_name)
    print(f"f_name: {f_name} type: {type(f_name)}")
    # parser.print_help()
    pass


def main():
    try:
        # read file and check the content 
        read_dataset()
        # df = pd.read_csv('./datasets/dataset_test.csv')
        # print(df.info())
        # print(df.describe())
    except Exception as e:
        print(f"Error: {e}")
        pass

if __name__ == "__main__":
    main()



# 1. all is numeric value (float , int) , wa can get all features
#     * dont  count the nan value 
# 2. all is null, we give result nan, count is 0
# 3. there is string , we don't take  account this columns  
