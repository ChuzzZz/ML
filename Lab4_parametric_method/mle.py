#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""module description"""

__author__ = 'CHUZ'

import matplotlib.pyplot as plt
import numpy as np


def gen_x():
    c_mean = np.array([[1, 1], [4, 4], [8, 1]])
    c_cov = 2 * np.eye(2)
    # Generate a uniform random sample from np.arange(3) of size 1000
    ur_sample = np.random.choice(3, 1000)
    ur_counts = np.bincount(ur_sample)

    x1 = np.random.multivariate_normal(c_mean[0], c_cov, ur_counts[0])
    x2 = np.random.multivariate_normal(c_mean[1], c_cov, ur_counts[1])
    x3 = np.random.multivariate_normal(c_mean[2], c_cov, ur_counts[2])
    return x1, x2, x3


def gen_xquote():
    c_mean = np.array([[1, 1], [4, 4], [8, 1]])
    c_cov = 2 * np.eye(2)
    # Generate a non-uniform random sample from np.arange(3) of size 1000
    nur_sample = np.random.choice(3, 1000, p=[0.8, 0.1, 0.1])
    nur_counts = np.bincount(nur_sample)

    xq1 = np.random.multivariate_normal(c_mean[0], c_cov, nur_counts[0])
    xq2 = np.random.multivariate_normal(c_mean[1], c_cov, nur_counts[1])
    xq3 = np.random.multivariate_normal(c_mean[2], c_cov, nur_counts[2])
    return xq1, xq2, xq3


def show_graph(set1, set2):
    x1, y1 = set1.T
    x2, y2 = set2.T
    plt.plot(x1, y1, 'b.')
    plt.plot(x2, y2, 'g.')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.show()


def likelihood(*c):
    class_num = len(c)
    for i in range(class_num):
        print('Class', i, 'mean : ', np.mean(c[i], axis=0))
        print('         cov : ', np.cov(c[i].T))


def bayes_classify(x, *c):
    class_num = len(c)
    pre = [c[i].shape[0] / 1000 for i in range(class_num)]
    class_mean = np.zeros((class_num, 2))  # n*2
    class_cov = np.zeros((class_num, 2, 2))  # n*2*2
    det = np.zeros((class_num, 1))
    # row n is the possibility of classify the sample to class n
    g = np.zeros((class_num, 1))

    for i in range(class_num):
        class_mean[i] = np.mean(c[i], axis=0)
        class_cov[i] = np.cov(c[i].T)
        det[i] = np.linalg.det(class_cov[i])
        g[i] = -0.5 * (x - class_mean[i]).dot(np.linalg.inv(class_cov[i])).dot(
            (x - class_mean[i])[np.newaxis].T) - 0.5 * np.log(det[i]) + np.log(pre[i])
    return np.argmax(g)


def euclid_classify(x, *c):
    class_num = len(c)
    class_mean = np.zeros((class_num, 2))  # n*2
    class_cov = np.zeros((class_num, 2, 2))  # n*2*2
    # row n is the possibility of classify the sample to class n
    g = np.zeros((class_num, 1))

    for i in range(class_num):
        class_mean[i] = np.mean(c[i], axis=0)
        class_cov[i] = np.cov(c[i].T)
        g[i] = -0.5 * (x - class_mean[i]).dot(np.linalg.inv(class_cov[i])).dot(
            (x - class_mean[i])[np.newaxis].T)

    return np.argmax(g)


def cal_accuracy_rate(classify_func, *c):
    class_num = len(c)
    set_size = 0
    accurate_num = 0
    for i in range(class_num):
        for j in range(c[i].shape[0]):
            set_size += 1
            if classify_func(c[i][j], *c) == i:
                accurate_num += 1
    print(accurate_num / set_size)


if __name__ == '__main__':
    X1, X2, X3 = gen_x()
    X_quote1, X_quote2, X_quote3 = gen_xquote()
    likelihood(X1, X2, X3)
    X = np.vstack((X1, X2, X3))
    X_quote = np.vstack((X_quote1, X_quote2, X_quote3))
    cal_accuracy_rate(bayes_classify, X1, X2, X3)
    cal_accuracy_rate(euclid_classify, X1, X2, X3)
    cal_accuracy_rate(bayes_classify, X_quote1, X_quote2, X_quote3)
    cal_accuracy_rate(euclid_classify, X_quote1, X_quote2, X_quote3)
    # show_graph(X, X_quote)
