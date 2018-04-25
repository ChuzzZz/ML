#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""module description"""

__author__ = 'CHUZ'

import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

alpha = 0.3
num_iter = 500


def load_data():
    boston = load_boston()
    X = boston.data
    y = boston.target
    return X, y


def feature_normalize(X):
    """Normalize features.

    :param X: ndarray of shape(m, n), m is the number of samples, n is the number of features
    :return: ndarray of shape(m, n), normalized X
    """
    m = X.shape[0]
    mu = np.mean(X, axis=0)  # column vector, shape=(n, 1)
    sigma = np.std(X, axis=0)
    mean_mat = np.tile(mu.T, (m, 1))  # shape=(m, n)
    std_mat = np.tile(sigma.T, (m, 1))
    X_norm = (X - mean_mat) / std_mat  # shape=(m, n)
    return X_norm


def compute_cost(X, y, theta):
    """Given theta, Compute the cost of dataset.

    :param X: ndarray of shape(m, n+1), m is the number of samples, n is the number of features
    :param y: ndarray of shape(m, )
    :param theta: estimate parameter, ndarray of shape(n+1, )
    :return: cost
    """
    m = y.size
    J = 0

    temp = X @ theta - y
    temp.shape = (m, 1)
    J = temp.T @ temp / (2 * m)  # J shape=(1, 1)
    return J[0][0]


def gradient_descent(X, y, theta, alpha, num_iter):
    """Run gradient descent.

    :param X: ndarray of shape(m, n+1), m is the number of samples, n is the number of features
    :param y: ndarray of shape(m, )
    :param theta: estimate parameter
    :param alpha: learning rate
    :param num_iter: num of iteration
    :return: optimal theta
    """
    m = y.size
    J_history = np.zeros((num_iter, 1))  # shape = (num_iter,1)
    for i in range(num_iter):
        theta = theta - X.T.dot(X.dot(theta) - y) / m * alpha
        J_history[i] = compute_cost(X, y, theta)
        print(J_history[i])
    return theta


if __name__ == '__main__':
    X, y = load_data()
    X = feature_normalize(X)
    # Add intercept term to X
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    theta = np.zeros((14,))
    gradient_descent(X, y, theta, alpha, num_iter)
