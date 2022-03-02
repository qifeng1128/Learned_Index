#!usr/bin/python
# -*- coding: utf-8 -*-

# Main file for create test data
from enum import Enum
import numpy as np
import csv
import random

SIZE = 1000000
BLOCK_SIZE = 100

# 生成数据的分布可以为随机、伯努利、泊松、指数、均匀、对数均匀
class Distribution(Enum):
    LINEAR = 0
    RANDOM = 1
    BINOMIAL = 2
    POISSON = 3
    EXPONENTIAL = 4
    NORMAL = 5
    LOGNORMAL = 6

# store path
filePath = {
    Distribution.LINEAR: "linear.csv",
    Distribution.RANDOM: "random.csv",
    Distribution.BINOMIAL: "binomial.csv",
    Distribution.POISSON: "poisson.csv",
    Distribution.EXPONENTIAL: "exponential.csv",
    Distribution.NORMAL: "normal.csv",
    Distribution.LOGNORMAL: "lognormal.csv"
}

# create data
def create_data(distribution, data_size=SIZE):
    if distribution == Distribution.LINEAR:
        weight = 1.0
        bias = 4.0
        data = [weight * i + bias for i in range(data_size)]
    elif distribution == Distribution.RANDOM:
        data = random.sample(range(data_size * 2), data_size)
    elif distribution == Distribution.BINOMIAL:
        data = np.random.binomial(100, 0.5, size=data_size)
    elif distribution == Distribution.POISSON:
        data = np.random.poisson(6, size=data_size)
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

if __name__ == "__main__":
    create_data(Distribution.EXPONENTIAL)
    create_data(Distribution.LINEAR)
    create_data(Distribution.LOGNORMAL)
    create_data(Distribution.NORMAL)
    create_data(Distribution.RANDOM)
