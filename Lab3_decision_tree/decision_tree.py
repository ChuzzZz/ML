#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""module description"""

__author__ = 'CHUZ'

import csv
import operator
import math


def load_data(filename):
    with open(filename, newline='', encoding='gbk') as f:
        reader = csv.reader(f)
        sample_list = list(reader)
        features = sample_list.pop(0)
        # delete feature 1 (i.e. No. feature)
        features.pop(0)
        features.pop(-1)
        for sample in sample_list:
            sample.pop(0)
            # sample_list m * n+1
            # features n * 1
    return sample_list, features


def cal_entropy(sample_list):
    """Compute the entropy of sample list.

    :param sample_list:
    :return: E(D), float
    """
    m = len(sample_list)
    label_count = {}
    for sample in sample_list:
        label = sample[-1]
        if label in label_count:
            label_count[label] += 1
        else:
            label_count[label] = 1
    entropy = 0.0
    for key in label_count:
        prob = label_count[key] / m
        entropy -= prob * math.log(prob, 2)
    return entropy


def split_samples(sample_list, axis, value):
    """Split samples according to specific axis and value.

    :param sample_list: a list of samples
    :param axis: feature index
    :param value: feature value
    :return:
    """
    if axis == len(sample_list[0]) - 1:
        raise ValueError('axis out of range in split_samples()!')
    ret_sample_list = []
    for sample in sample_list:
        if sample[axis] == value:
            new_sample = sample[:]
            new_sample.pop(axis)
            ret_sample_list.append(new_sample)
    return ret_sample_list


def cal_gain(sample_list, axis):
    """Calculate Gain(D, a).

    :param sample_list: a list of samples
    :param axis: feature index
    :return: info gained, float
    """
    if axis == len(sample_list[0]) - 1:
        raise ValueError('axis out of range in cal_gain()!')
    ent_parent = cal_entropy(sample_list)
    value_set = set([sample[axis] for sample in sample_list])  # set of values of the feature
    ent_axis = 0.0
    for value in value_set:
        sub_list = split_samples(sample_list, axis, value)
        proportion = len(sub_list) / len(sample_list)
        ent_axis += proportion * cal_entropy(sub_list)
    info_gain = ent_parent - ent_axis
    return info_gain


def get_best_feature_index(sample_list):
    """Choose the best feature for the samples.

    :param sample_list: list of samples
    :return:
    """
    feature_num = len(sample_list[0]) - 1
    gain_list = [cal_gain(sample_list, i) for i in range(feature_num)]
    return gain_list.index(max(gain_list))


def get_majority_label(label_list):
    """Get the label of maximum occurrence.

    :param label_list: list of labels
    :return:
    """
    label_count = {}
    for label in label_list:
        if label not in label_count:
            label_count[label] = 0
        label_count[label] += 1
    sorted_label_count = sorted(label_count.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sorted_label_count[0][0]


def create_tree(sample_list, features):
    """Create decision tree.

    :param sample_list:
    :param features:
    :return:
    """
    label_list = [sample[-1] for sample in sample_list]
    # all samples have the same label
    if label_list.count(label_list[0]) == len(label_list):
        return label_list[0]
    # no feature available
    if len(features) == 1:
        return get_majority_label(sample_list)
    # divide samples according to the best feature
    best_feature_index = get_best_feature_index(sample_list)
    best_feature = features[best_feature_index]
    my_tree = {best_feature: {}}
    best_feature_value_set = set([sample[best_feature_index] for sample in sample_list])
    for value in best_feature_value_set:
        sub_feature = features[:]
        sub_feature.pop(best_feature_index)
        my_tree[best_feature][value] = create_tree(split_samples(sample_list, best_feature_index, value), sub_feature)
    return my_tree


def classify(tree, features, test_sample):
    """Classify a test sample.

    :param tree:
    :param features:
    :param test_sample:
    :return:
    """
    feature = list(tree.keys())[0]
    subtree = tree[feature]
    feature_index = features.index(feature)
    for key in subtree:
        if test_sample[feature_index] == key:
            if type(subtree[key]).__name__ == 'dict':
                vote_label = classify(subtree[key], features, test_sample)
            else:
                vote_label = subtree[key]
    return vote_label


def compute_accuracy(decision_tree, features, test_samples):
    """Compute the decision accuracy on test samples.

    :param decision_tree:
    :param features:
    :param test_samples:
    :return:
    """
    num_sample = len(test_samples)
    num_correct = 0
    for i in range(num_sample):
        if test_samples[i][-1] == classify(decision_tree, features, test_samples[i]):
            num_correct += 1
    print(num_correct / num_sample)


if __name__ == '__main__':
    data, features = load_data('Watermelon-train1.csv')
    test_data, features = load_data('Watermelon-test1.csv')
    d_tree = create_tree(data, features)
    print(d_tree)
    compute_accuracy(d_tree, features, test_data)
