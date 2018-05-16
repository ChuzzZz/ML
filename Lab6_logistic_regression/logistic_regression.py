#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""module description"""

__author__ = 'CHUZ'

import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

alpha = 1
my_lambda = 0


def load_data(file_path):
    """

    :param file_path:
    :return: X of shape(m, n), y of shape(m, )
    """
    with open(file_path, newline='') as f:
        my_data = np.loadtxt(f, delimiter=',')
    X = my_data[:, 1:]
    y = my_data[:, 0] - 1
    return X, y


def sigmoid(Z):
    return 1.0 / (1.0 + np.exp(-Z))


def scale_feature(X):
    """Feature scaling.

    :param X: ndarray of shape(m, n)
    :return: scaled X, ndarray of shape(m, n+1)
    """
    m = X.shape[0]

    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    X_scaled = np.concatenate((np.ones((m, 1)), X_scaled), axis=1)  # add bias term
    return X_scaled


def compute_cost(X, y, theta, my_lambda):
    """Given theta and lambda, compute cost.

    :param X: ndarray of shape(m, n+1)
    :param y: ndarray of shape(m, )
    :param theta: ndarray of shape(n+1, )
    :param my_lambda: lambda
    :return: cost
    """
    m = y.size
    J = 0

    h = sigmoid(X @ theta)  # shape = (m, )
    L2 = my_lambda * np.sum(theta[1:] ** 2) / 2
    J = -(y @ np.log(h) + (1 - y) @ np.log(1 - h)) + L2
    J /= m
    return J


def gradient_descent(X, y, theta, alpha, my_lambda=0, max_iter=500, tol=1e-4):
    """Optimize theta.

    :param X: ndarray of shape(m, n+1)
    :param y: ndarray of shape(m, )
    :param theta: ndarray of shape(n+1, )
    :param alpha: learning rate
    :param my_lambda: lambda
    :param max_iter: maximum iteration
    :return: optimal theta
    """
    m = y.size
    J_history = np.zeros((max_iter,))

    for i in range(max_iter):
        h = sigmoid(X @ theta)  # shape = (m, )
        grad = my_lambda * theta
        grad[0] = 0
        grad += X.T @ (h - y) / m
        theta -= alpha * grad
        J_history[i] = compute_cost(X, y, theta, my_lambda)
        if i > 0 and abs(J_history[i] - J_history[i - 1]) < tol:
            break

    return theta


def predict(X, theta):
    """Using optimal theta to predict X's class.

    :param X: ndarray of shape(m, n+1)
    :param theta: ndarray of shape(n+1, )
    :return:
    """
    m = X.shape[0]
    pred = np.ones((m,))
    h = sigmoid(X @ theta)
    pred[h < 0.5] = 0
    return pred


def one_vs_all(X, y, num_labels, alpha, my_lambda, max_iter=150, tol=1e-4):
    """Compute theta for every class.

    :param X: ndarray of shape(m, n+1)
    :param y: ndarray of shape(m, )
    :param num_labels: number of classes
    :param alpha: learning rate
    :param my_lambda: lambda
    :param max_iter: maximum iteration
    :param tol: tolerance
    :return: theta_mat, ndarray of shape(num_labels, n+1)
    """
    m, n = X.shape
    theta_mat = np.zeros((num_labels, n))
    for i in range(num_labels):
        initial_theta = np.zeros((n,))
        theta = gradient_descent(X, (y == i), initial_theta, alpha, my_lambda)
        theta_mat[i, :] = theta
    return theta_mat


def double_classification():
    X, y = load_data('wine_binary.csv')
    m, n = X.shape
    print('Double-classification: {} samples, {} features'.format(m, n))
    X = scale_feature(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    theta = np.zeros((14,))
    optimal_theta = gradient_descent(X_train, y_train, theta, alpha, my_lambda)
    prediction = predict(X_test, optimal_theta)
    accuracy = np.sum(prediction == y_test) / y_test.size
    print('accuracy: {0:.1f}%'.format(accuracy * 100))


def multi_classification():
    X, y = load_data('wine.csv')
    num_labels = np.unique(y).size  # number of different classes
    m, n = X.shape
    print('Multi-classification: {} samples, {} features'.format(m, n))
    X = scale_feature(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    all_theta = one_vs_all(X_train, y_train, num_labels, alpha, my_lambda)
    prediction = np.argmax(X_test @ all_theta.T, axis=1)
    accuracy = np.sum(prediction == y_test) / y_test.size
    print('accuracy: {}%'.format(accuracy * 100))


if __name__ == '__main__':
    double_classification()
    multi_classification()
