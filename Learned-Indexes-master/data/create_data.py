#!usr/bin/python
# -*- coding: utf-8 -*-

# Main file for create test data
from enum import Enum
import numpy as np
import csv
import random
import pandas as pd

SIZE = 10000000
BLOCK_SIZE = 100

# 生成数据的分布可以为随机、指数、正态、对数正态
class Distribution(Enum):
    LINEAR = 0
    RANDOM = 1
    EXPONENTIAL = 2
    NORMAL = 3
    LOGNORMAL = 4
    WIKI = 5
    OSM = 6
    BOOKS = 7
    FB = 8

# store path
filePath = {
    Distribution.LINEAR: "linear.csv",
    Distribution.RANDOM: "random.csv",
    Distribution.EXPONENTIAL: "exponential.csv",
    Distribution.NORMAL: "normal.csv",
    Distribution.LOGNORMAL: "lognormal.csv",
    Distribution.WIKI: "data/wiki_ts_200M_uint64.csv"
}

# create data
def create_data(distribution, data_size = SIZE):
    if distribution == Distribution.LINEAR:
        random_data = random.sample(range(data_size * 2), data_size)
        weight = 1.0
        bias = 0.0
        data = [weight * random_data[i] + bias for i in range(data_size)]
    elif distribution == Distribution.RANDOM:
        data = random.sample(range(data_size * 2), data_size)
    elif distribution == Distribution.EXPONENTIAL:
        data = np.random.exponential(10, size=data_size)
    elif distribution == Distribution.LOGNORMAL:
        data = np.random.lognormal(0, 2, data_size)
    else:
        data = np.random.normal(1000, 100, size=data_size)
    res_path = filePath[distribution]
    data.sort()
    with open(res_path, 'w') as csvFile:
        csv_writer = csv.writer(csvFile)
        i = 0
        for d in data:
            csv_writer.writerow([d, 1.0 * i / BLOCK_SIZE])
            i += 1


def create_data_no_storage(distribution, data_size = SIZE):
    if distribution == Distribution.LINEAR:
        weight = 1.0
        bias = 4.0
        data = [weight * i + bias for i in range(data_size)]
    elif distribution == Distribution.RANDOM:
        data = random.sample(range(data_size * 2), data_size)
    elif distribution == Distribution.EXPONENTIAL:
        data = np.random.exponential(10, size=data_size)
    elif distribution == Distribution.LOGNORMAL:
        data = np.random.lognormal(0, 2, data_size)
    elif distribution == Distribution.WIKI:
        data = pd.read_csv(filePath[distribution], header=None)
        data = data.iloc[:, 0]
    else:
        data = np.random.normal(1000, 100, size=data_size)
    return data

if __name__ == "__main__":
    create_data(Distribution.EXPONENTIAL)
    create_data(Distribution.LINEAR)
    create_data(Distribution.LOGNORMAL)
    create_data(Distribution.NORMAL)
    create_data(Distribution.RANDOM)
