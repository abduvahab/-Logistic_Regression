import pandas as pd
from matplotlib import pyplot as plt
import math


def get_cours_list(df: pd.DataFrame):
    column_list = df.columns
    course_list = []
    not_course = [
        'Index', 'Hogwarts House', 'First Name',
        'Last Name', 'Birthday', 'Best Hand'
    ]
    for col in column_list:
        if col not in not_course:
            course_list.append(col)
    # print(course_list)
    return course_list


def create_correlation_table(course_list: list):
    cor_table = pd.DataFrame(index=course_list, columns=course_list)
    # print(cor_table)
    return cor_table


def find_correlation(df, cor_table):
    m = len(df)
    for col in cor_table.columns:
        col_data = df[col]
        col_mean = col_data.mean()
        for row in cor_table.index:
            if col == row:
                continue
            row_data = df[row]
            row_mean = row_data.mean()
            Numerator = 0
            D_col = 0
            D_row = 0
            for i in range(m):
                mlti_ab = ((col_data[i] - col_mean)*(row_data[i] - row_mean))
                Numerator += mlti_ab
                D_col += (col_data[i] - col_mean)**2
                D_row += (row_data[i] - row_mean)**2
            D_all = math.sqrt(D_col * D_row)
            r = Numerator / D_all
            cor_table.loc[row, col] = r
    print(cor_table)
    return cor_table


def main():
    try:
        # read dataset
        f_name = "./datasets/dataset_train.csv"
        df = pd.read_csv(f_name)
        course_list = get_cours_list(df)
        df = df.dropna()
        df.reset_index(drop=True, inplace=True)
        cor_table = create_correlation_table(course_list)
        cor_table = find_correlation(df, cor_table)
        max_corr = cor_table.max().max()
        max_loc = cor_table.stack().idxmax()
        max_row, max_col = max_loc

        min_corr = cor_table.min().min()
        min_loc = cor_table.stack().idxmin()
        min_row, min_col = min_loc

        if abs(min_corr) > abs(max_corr):
            print(f"two features most similaire are: {min_row}, {min_col}")
            plt.scatter(df[min_row], df[min_col])
            plt.xlabel(min_row)
            plt.ylabel(min_col)
        else:
            print(f"two features most similaire are: {max_row} and {max_col}")
            plt.scatter(df[max_row], df[max_col])
            plt.xlabel(max_row)
            plt.ylabel(max_col)
        plt.show()
        # print(df.info())
        # m = len(course_list)
        # for i in range(m-1):
        #     for j in range(i + 1, m):
        #         plt.scatter(df[course_list[i]], df[course_list[j]])
        #         plt.xlabel(course_list[i])
        #         plt.ylabel(course_list[j])
        #         plt.show()
    except Exception as e:
        print(f"Error:{e}")
    pass


if __name__ == "__main__":
    main()
