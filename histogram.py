import pandas as pd
from matplotlib import pyplot as plt


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


def create_std_table(course_list: list, house_list: list) -> pd.DataFrame:
    # std_df = pd.DataFrame({"course_list" : course_list})
    # for h in house_list:
    #     std_df[h] = np.nan
    # std_df["av_std"] = np.nan
    # std_df.index = course_list
    columns = house_list.copy()
    columns.append("av_std")
    std_df = pd.DataFrame(index=course_list, columns=columns)
    # print(std_df)
    return std_df


def fill__std_table(std_df, df, house_list) -> pd.DataFrame:
    for index in std_df.index:
        stds = df.groupby(['Hogwarts House'])[index].std()
        for house in house_list:
            std_df.at[index, house] = stds.at[house]
        std_df.at[index, 'av_std'] = stds.mean()
    print(std_df)
    return std_df


def main():
    # read data set
    f_name = "./datasets/dataset_train.csv"
    df = pd.read_csv(f_name)
    # get the course list
    course_list = get_cours_list(df)
    # get the house list
    house_list = df['Hogwarts House'].unique().tolist()
    # cretae a table std of each house for every course
    std_df = create_std_table(course_list, house_list)
    # find the std of each house for course
    std_df = fill__std_table(std_df, df, house_list)
    stable_course = std_df['av_std'].idxmin()
    print("*" * 50)
    msg = (
        "the course most homogeneos"
        " score distribition accross all Hogwarts house"
    )
    print(f"{msg}: {stable_course}")
    df.hist(stable_course, 'Hogwarts House')
    plt.show()


if __name__ == "__main__":
    main()
