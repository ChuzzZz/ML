#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""module description"""

import csv
import operator

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold

__author__ = 'CHUZ'


def img2rvector(raw_data):
    data = [float(i) for i in raw_data.split(' ')[:256]]
    vector = np.array([data])    # 16x16 img to 1x256 vector

    label = -1
    number = raw_data.split(' ')[256:]
    for idx, val in enumerate(number):
        if val == '1':
            label = idx
    return vector, label


def get_training_data():
    train_matrix = np.zeros((1, 256))
    train_labels = []
    with open('semeion_train.csv', newline='') as f:
        data_reader = csv.reader(f)

        # train_labels to nx1 vector
        # train_matrix to nx256 matrix
        # n is the size of training examples
        for row in data_reader:
            img_data = img2rvector(row[0])
            train_matrix = np.row_stack((train_matrix, img_data[0]))
            train_labels.append(img_data[1])
        train_matrix = np.delete(train_matrix, 0, 0)
        train_labels = np.array(train_labels)
    return train_matrix, train_labels


def get_test_data():
    testX = np.zeros((1, 256))
    test_labels = []

    # test_labels to nx1 vector
    # testX to nx256 matrix
    # n is the size of test examples
    with open('semeion_test.csv') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            img_data = img2rvector(row[0])
            testX = np.row_stack((testX, img_data[0]))
            test_labels.append(img_data[1])
        testX = np.delete(testX, 0, 0)
        test_labels = np.array(test_labels)
    return testX, test_labels


def get_all_data():
    X = np.zeros((1, 256))
    labels = []

    with open('semeion_train.csv', newline='') as f:
        data_reader = csv.reader(f)

        for row in data_reader:
            img_data = img2rvector(row[0])
            X = np.row_stack((X, img_data[0]))
            labels.append(img_data[1])
    with open('semeion_test.csv') as file:
        data_reader = csv.reader(file)
        for row in data_reader:
            img_data = img2rvector(row[0])
            X = np.row_stack((X, img_data[0]))
            labels.append(img_data[1])

    X = np.delete(X, 0, 0)
    labels = np.array(labels)
    return X, labels


def classify(inX, train_matrix, train_labels, k):
    train_num = train_matrix.shape[0]
    sq_diffmat = (np.tile(inX, (train_num, 1)) - train_matrix) ** 2
    square_distance = sq_diffmat.sum(axis=1)
    distances = np.sqrt(square_distance)
    indices = distances.argsort()

    class_count = {}
    for i in range(k):
        vote_label = train_labels[indices[i]]
        if vote_label in class_count:
            class_count[vote_label] += 1
        else:
            class_count[vote_label] = 1

    sorted_class_count = sorted(
        class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def cal_accuracy_rate(trainX, train_labels, testX, test_labels, k):
    score = 0.0
    n = testX.shape[0]
    for i in range(n):
        vote_label = classify(testX[i], trainX, train_labels, k)
        score += (test_labels[i] == vote_label)
    accuracy_rate = score/n
    return accuracy_rate


def handwriting_test():
    train_data = get_training_data()
    test_data = get_test_data()
    for k in [1, 3, 5, 10]:
        print('k =', k, end=' ')
        print('accuracy =', cal_accuracy_rate(train_data[0], train_data[1], test_data[0], test_data[1], k))


def crossvali_handwriting_test():
    # 画图用
    x_list = list(range(1, 11))
    y_list = []

    X, labels = get_all_data()
    kf = KFold(n_splits=5)
    for k in range(1, 11):
        sum_accuracy = 0.0
        for train_index, test_index in kf.split(X):
            # print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X[train_index], X[test_index]
            labels_train, labels_test = labels[train_index], labels[test_index]
            sum_accuracy += cal_accuracy_rate(X_train,
                                              labels_train, X_test, labels_test, k)
        mean_accuracy = sum_accuracy / 5
        print('k =', k, end=' ')
        print('accuracy =', mean_accuracy)
        y_list.append(mean_accuracy)

    # 绘图
    plt.figure('Line fig')
    ax = plt.gca()
    ax.set_xlabel('k')
    ax.set_ylabel('accuracy rate')
    ax.plot(x_list, y_list, color='r', linewidth=1, alpha=0.6)
    plt.show()

if __name__ == '__main__':
    handwriting_test()
    crossvali_handwriting_test()
    
