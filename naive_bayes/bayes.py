#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""module description"""

import csv
import math

import numpy as np

__author__ = 'CHUZ'


def load_data(filename):
    with open(filename, newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)[1:]
        for i in range(len(rows)):
            rows[i] = [float(x) for x in rows[i]][1:]
        dataset = np.array(rows)
    return dataset


def pre_probability(train_set):
    pos_prob = np.sum(train_set, axis=0)[-1] / train_set.shape[0]
    neg_prob = 1.0 - pos_prob
    return pos_prob, neg_prob


def pre_probability_lap(train_set):
    pos_prob = (np.sum(train_set, axis=0)[-1] + 1) / (train_set.shape[0] + 2)
    neg_prob = 1.0 - pos_prob
    return pos_prob, neg_prob


def condition_probability(train_set, test):
    feature_num = train_set.shape[1] - 1
    pos_data = train_set[train_set[:, -1] == 1.0, :]
    pos_num = pos_data.shape[0]
    neg_data = train_set[train_set[:, -1] == 0.0, :]
    neg_num = neg_data.shape[0]

    cond_result = np.zeros((feature_num, 2))
    for i in range(feature_num):
        cond_result[i, 0] = np.sum(pos_data[:, i] == test[i]) / pos_num
        cond_result[i, 1] = np.sum(neg_data[:, i] == test[i]) / neg_num
    for i in range(6, 8):
        pos_mean = np.mean(pos_data[:, i])
        pos_std = np.std(pos_data[:, i])
        neg_mean = np.mean(neg_data[:, i])
        neg_std = np.std(neg_data[:, i])
        cond_result[i, 0] = 1.0 / (math.sqrt(2 * np.pi) * pos_std) * np.exp(
            -1 * (test[i] - pos_mean) ** 2 / (2 * pos_std ** 2))
        cond_result[i, 1] = 1.0 / (math.sqrt(2 * np.pi) * neg_std) * np.exp(
            -1 * (test[i] - neg_mean) ** 2 / (2 * neg_std ** 2))

    return cond_result


def condition_probability_lap(train_set, test):
    feature_num = train_set.shape[1] - 1
    attr_num = [3, 3, 3, 3, 3, 2, 0, 0]
    pos_data = train_set[train_set[:, -1] == 1.0, :]
    pos_num = pos_data.shape[0]
    neg_data = train_set[train_set[:, -1] == 0.0, :]
    neg_num = neg_data.shape[0]

    cond_result = np.zeros((feature_num, 2))
    for i in range(feature_num):
        cond_result[i, 0] = (np.sum(pos_data[:, i] == test[i]) + 1) / (pos_num + attr_num[i])
        cond_result[i, 1] = (np.sum(neg_data[:, i] == test[i]) + 1) / (neg_num + attr_num[i])
    for i in range(6, 8):
        pos_mean = np.mean(pos_data[:, i])
        pos_std = np.std(pos_data[:, i])
        neg_mean = np.mean(neg_data[:, i])
        neg_std = np.std(neg_data[:, i])
        cond_result[i, 0] = 1.0 / (math.sqrt(2 * np.pi) * pos_std) * np.exp(
            -1 * (test[i] - pos_mean) ** 2 / (2 * pos_std ** 2))
        cond_result[i, 1] = 1.0 / (math.sqrt(2 * np.pi) * neg_std) * np.exp(
            -1 * (test[i] - neg_mean) ** 2 / (2 * neg_std ** 2))

    return cond_result


def classify(cond_result, pre_prob):
    pos_result = pre_prob[0]
    neg_result = pre_prob[1]
    for i in range(cond_result.shape[0]):
        pos_result *= cond_result[i, 0]
        neg_result *= cond_result[i, 1]
    return 1 if pos_result > neg_result else 0


def classify_lap(cond_result, pre_prob):
    pos_result = math.log2(pre_prob[0])
    neg_result = math.log2(pre_prob[1])
    result = np.sum(np.log2(cond_result), axis=0)
    pos_result += result[0]
    neg_result += result[1]
    return 1 if pos_result > neg_result else 0


def cal_accuracy_rate(train_set, test_set):
    pre_prob = pre_probability(train_set)
    test_class = test_set[:, -1]
    vote_class = np.linspace(2, 3, test_class.size)
    for i in range(test_class.size):
        cond_prob = condition_probability(train_set, test_set[i])
        vote_class[i] = classify(cond_prob, pre_prob)
        print(vote_class[i])

    accuracy_rate = np.sum(test_class == vote_class) / test_set.shape[0]
    return accuracy_rate


def cal_accuracy_rate_lap(train_set, test_set):
    pre_prob = pre_probability_lap(train_set)
    test_class = test_set[:, -1]
    vote_class = np.linspace(2, 3, test_class.size)
    for i in range(test_class.size):
        cond_prob = condition_probability_lap(train_set, test_set[i])
        vote_class[i] = classify_lap(cond_prob, pre_prob)
        print(vote_class[i])

    accuracy_rate = np.sum(test_class == vote_class) / test_set.shape[0]
    return accuracy_rate


if __name__ == '__main__':
    train_data = load_data('train.csv')
    test_data = load_data('test.csv')
    # pre_p = pre_probability(train_data)
    # test1 = np.array([3,1,1,1,1,1,0.556,0.215,1])
    # cond_p = condition_probability(train_data, test1)
    # print(classify(cond_p, pre_p))
    ar = cal_accuracy_rate(train_data, test_data)
    print(ar)
    ar_l = cal_accuracy_rate_lap(train_data, test_data)
    print(ar_l)
