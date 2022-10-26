import pandas as pd

# 用于ALEX索引读取的数据
def change_generate_data(input_path,output_path):
    data = pd.read_csv(input_path, header=None)
    # 从总数据中抽取第一列的值放入txt文件里
    with open(output_path, 'w') as f:
        for i in range(data.shape[0]):
            f.write(str(data.iloc[i, 0]))
            f.write("\n")


if __name__ == "__main__":
    # exponential
    change_generate_data("D:/lunwen/Learned-Indexes-master/data/exponential.csv",
                         "D:/lunwen/ALEX-master/resources/exponential.txt")
    # linear
    change_generate_data("D:/lunwen/Learned-Indexes-master/data/linear.csv",
                         "D:/lunwen/ALEX-master/resources/linear.txt")
    # lognormal
    change_generate_data("D:/lunwen/Learned-Indexes-master/data/lognormal.csv",
                         "D:/lunwen/ALEX-master/resources/lognormal.txt")
    # normal
    change_generate_data("D:/lunwen/Learned-Indexes-master/data/normal.csv",
                         "D:/lunwen/ALEX-master/resources/normal.txt")
    # random
    change_generate_data("D:/lunwen/Learned-Indexes-master/data/random.csv",
                         "D:/lunwen/ALEX-master/resources/random.txt")
    # book
    change_generate_data("D:/lunwen/Learned-Indexes-master/data/books_200M_uint64.csv",
                         "D:/lunwen/ALEX-master/resources/books_200M_uint64.txt")
    # fb
    change_generate_data("D:/lunwen/Learned-Indexes-master/data/fb_200M_uint64.csv",
                         "D:/lunwen/ALEX-master/resources/fb_200M_uint64.txt")
    # osm
    change_generate_data("D:/lunwen/Learned-Indexes-master/data/osm_cellids_200M_uint64.csv",
                         "D:/lunwen/ALEX-master/resources/osm_cellids_200M_uint64.txt")
    # wiki
    change_generate_data("D:/lunwen/Learned-Indexes-master/data/wiki_ts_200M_uint64.csv",
                         "D:/lunwen/ALEX-master/resources/wiki_ts_200M_uint64.txt")

