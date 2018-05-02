#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""module description"""

__author__ = 'CHUZ'

import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut

alpha = 0.3
num_iter = 1000
my_lambda = 7.0


def load_data():
    boston = load_boston()
    X = boston.data
    y = boston.target
    return X, y


def scale_feature(X):
    """Feature scaling.

    :param X: ndarray of shape(m, n), m is the number of samples, n is the number of features
    :return: ndarray of shape(m, n), scaled X
    """
    m = X.shape[0]
    mu = np.mean(X, axis=0)  # column vector, shape=(n, 1)
    sigma = np.std(X, axis=0)
    mean_mat = np.tile(mu.T, (m, 1))  # shape=(m, n)
    std_mat = np.tile(sigma.T, (m, 1))
    X_scaled = (X - mean_mat) / std_mat  # shape=(m, n)
    return X_scaled


def compute_cost(X, y, theta, my_lambda=0):
    """Given theta, Compute the cost of dataset.

    :param X: ndarray of shape(m, n+1), m is the number of samples, n is the number of features
    :param y: ndarray of shape(m, )
    :param theta: estimate parameter, ndarray of shape(n+1, )
    :param my_lambda: lambda
    :return: cost
    """
    m = y.size
    J = 0

    regularization_term = my_lambda / (2 * m) * sum(theta[1:] ** 2)
    J = sum((X @ theta - y) ** 2) / (2 * m) + regularization_term  # shape=(1, 1)
    return J


def gradient_descent(X, y, theta, alpha, my_lambda=0, num_iter=200):
    """Run gradient descent.

    :param X: ndarray of shape(m, n+1), m is the number of samples, n is the number of features
    :param y: ndarray of shape(m, )
    :param theta: estimate parameter
    :param alpha: learning rate
    :param my_lambda: lambda
    :param num_iter: num of iteration
    :return: optimal theta
    """
    m = y.size
    J_history = np.zeros((num_iter,))  # shape = (num_iter, )

    for i in range(num_iter):
        theta_grad = theta * my_lambda / m
        theta_grad[0] = 0
        theta_grad = alpha * (theta_grad + X.T @ (X @ theta - y) / m)
        theta -= theta_grad
        J_history[i] = compute_cost(X, y, theta, my_lambda)
    # print(J_history[-6:-1])
    return theta


def choose_best_lambda(X, y, lambda_vec=[0.1, 1.0, 10.0]):
    m, n = X.shape
    loo = LeaveOneOut()
    validation_error = np.zeros((len(lambda_vec), ))

    for i in range(len(lambda_vec)):
        for train_index, vali_index in loo.split(X):
            X_train, X_vali = X[train_index], X[vali_index]
            y_train, y_vali = y[train_index], y[vali_index]
            theta = np.zeros((n, ))
            theta = gradient_descent(X_train, y_train, theta, alpha=alpha, my_lambda=lambda_vec[i],  num_iter=500)
            validation_error[i] += predict(X_vali, y_vali, theta)
        print(validation_error[i] / m)
    validation_error /= m
    min_index = np.argmin(validation_error)
    return lambda_vec[min_index]


def predict(X, y, theta):
    """RMSE on test set.

    :param X:
    :param y:
    :param theta:
    :return:
    """
    RMSE = 0

    RMSE = np.sqrt(np.mean((X @ theta - y) ** 2))
    return RMSE


if __name__ == '__main__':
    X, y = load_data()
    X = scale_feature(X)
    # Add bias term to X
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # best_lambda = choose_best_lambda(X_train, y_train, [0.1, 0.3, 1.0, 3.0, 5.0, 7.0, 8.5, 10.0, 12.0])  # best_lambda = 7.0
    theta = np.zeros((14,))
    theta = gradient_descent(X_train, y_train, theta, alpha, my_lambda, num_iter)
    RMSE = predict(X_test, y_test, theta)
    print(RMSE)
