from ann import ANN
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold

import os
import csv
import time
import util
import argparse
import numpy as np


def main(args):
    X = []
    y = []

    with open(os.path.abspath(args.data)) as data_file:
        data_reader = csv.reader(data_file)
        for row in data_reader:
            X.append([float(x) for x in row])

    with open(os.path.abspath(args.labels)) as labels_file:
        labels_reader = csv.reader(labels_file)
        for row in labels_reader:
            y.append(int(row[0]))

    X = np.array(X)
    y = np.array(y)

    X_resampled, y_resampled = SMOTE().fit_resample(X, y)
    print('Classes:')
    print(Counter(y))
    print('After Resampling:')
    print(Counter(y_resampled))

    skf = StratifiedKFold(n_splits=2)
    skf.get_n_splits(X_resampled, y_resampled)

    for train_index, test_index in skf.split(X_resampled, y_resampled):
        X_train, X_test = X_resampled[train_index], X_resampled[test_index]
        y_train, y_test = y_resampled[train_index], y_resampled[test_index]

    util.write_csv('training_set.csv', X_train)
    util.write_csv('training_labels.csv', np.transpose(
        np.array(y_train, ndmin=2)))
    util.write_csv('validation_set.csv', X_test)
    util.write_csv('validation_labels.csv', np.transpose(
        np.array(y_test, ndmin=2)))

    ann = ANN(
        args.n_in,
        args.n_h1,
        args.n_h2,
        args.n_out,
        args.eta,
        args.max_epoch
    )

    print('Training Phase..')
    ann.train(X_train, y_train)

    print('Testing Phase')
    ann.test(X_test, y_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for running ANN')

    parser.add_argument(
        '--data',
        dest='data',
        metavar='<path-to-data>',
        required=True,
        help='Path to the data file'
    )

    parser.add_argument(
        '--labels',
        dest='labels',
        metavar='<path-to-labels>',
        required=True,
        help='Path to the labels file'
    )

    parser.add_argument(
        '--in',
        dest='n_in',
        type=int,
        required=True,
        help='number of input neurons'
    )

    parser.add_argument(
        '--h1',
        dest='n_h1',
        type=int,
        required=True,
        help='number of neurons in 1st hidden layer'
    )

    parser.add_argument(
        '--h2',
        dest='n_h2',
        type=int,
        required=True,
        help='number of neurons in 2nd hidden layer'
    )

    parser.add_argument(
        '--out',
        dest='n_out',
        type=int,
        required=True,
        help='number of output neurons'
    )

    parser.add_argument(
        '--eta',
        dest='eta',
        type=float,
        required=True,
        help='learning rate'
    )

    parser.add_argument(
        '--max_epoch',
        dest='max_epoch',
        type=int,
        default=30000,
        help='Maximum epoch'
    )
    args = parser.parse_args()

    start_time = time.time()
    main(args)
    print("--- %s seconds ---" % (time.time() - start_time))
