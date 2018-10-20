from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import time
import argparse
import util
import numpy as np
import pandas as pd


def main(args):
    X = pd.read_csv(args.data)
    y = pd.read_csv(args.labels)

    X, y = SMOTE().fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    svclassifier = SVC(kernel='linear')
    svclassifier.fit(X_train, y_train)

    y_pred = svclassifier.predict(X_test)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    X_test = pd.read_csv(args.test_set)
    y_pred = svclassifier.predict(X_test)
    util.write_csv('predicted_svm.csv', np.transpose(
        np.array(y_pred, ndmin=2)))

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
        '--test-set',
        dest='test_set',
        metavar='<path-to-test-set>',
        required=True,
        help='Path to the test_set file'
    )

    args = parser.parse_args()

    start_time = time.time()
    main(args)
    print("--- %s seconds ---" % (time.time() - start_time))
