#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as pl


def build_matrix(x, nwindow):
    nx, nt = x.shape
    X = np.empty((nt - nwindow, nx * nwindow))
    for i in range(nt-nwindow):
        X[i, :] = x[:, i:i+nwindow].flatten()
    return X


def load_data(predict_id=0, train_ids=[1], train_frac=0.6, nwindow=5,
              self_offset=24, l2=0.1):
    # Load the target list and dataset.
    targets = map(int, open("targets.txt").read().splitlines())
    data = np.loadtxt("dataset_1.txt")
    target_ids = data[:, 0]

    # Linearly interpolate NaNs.
    data = data[:, 3:]
    for j, row in enumerate(data):
        inds = np.arange(len(row))[~np.isfinite(row)]
        for i in inds:
            data[j, i] = 0.5*(data[j, i-1] + data[j, i+1])

    # Normalize.
    data = (data-np.mean(data, axis=1)[:, None])/np.var(data, axis=1)[:, None]

    # Chunk up the data.
    predict_ids = target_ids == int(targets[predict_id])
    y = data[predict_ids, :]

    # Extract the flux time series from the training target(s).
    train_mask = np.sum([target_ids == int(targets[i]) for i in train_ids],
                        axis=0)
    x = data[train_mask, :]

    # Separate into test and train sets.
    train_number = int(train_frac * data.shape[-1])
    i0 = self_offset + nwindow
    y_train, y_test = y[:, i0:train_number-i0], y[:, train_number+i0:-i0]
    x_train, x_test = x[:, i0:train_number-i0], x[:, train_number+i0:-i0]

    # Build the x matrix.
    X_train = build_matrix(x_train, nwindow)
    X_test = build_matrix(x_test, nwindow)

    # Extract the lagged time series from the target series.
    X_train = np.concatenate((X_train,
                              build_matrix(y[:, :train_number-2*i0], nwindow),
                              build_matrix(y[:, 2*i0:train_number], nwindow),
                              np.ones((X_train.shape[0], 1))),
                             axis=1)
    X_test = np.concatenate((X_test,
                             build_matrix(y[:, train_number:-2*i0], nwindow),
                             build_matrix(y[:, train_number+2*i0:], nwindow),
                             np.ones((X_test.shape[0], 1))),
                            axis=1)

    # Trim the ends off the flux arrays.
    nw = int(np.floor(0.5 * nwindow))
    y_train, y_test = y_train[:, nw:-nw-1], y_test[:, nw:-nw-1]

    # Add L2 regularization.
    X_train = np.concatenate((X_train, l2 * np.ones((1, X_train.shape[1]))))
    y_train = np.concatenate((y_train, np.zeros((y_train.shape[0], 1))),
                             axis=1)

    X_test = np.concatenate((X_test, l2 * np.ones((1, X_test.shape[1]))))
    y_test = np.concatenate((y_test, np.zeros((y_test.shape[0], 1))), axis=1)

    return (X_train, y_train), (X_test, y_test)


if __name__ == "__main__":
    ind = 20

    (X_train, y_train), (X_test, y_test) = load_data()
    # y_train[ind] += 1e-4 * np.sin(np.arange(len(y_train[ind])) * 1e-2)
    y_test[ind, 50:100] -= 1e-4
    c, r, rank, s = np.linalg.lstsq(X_train, y_train[ind])

    print(np.sum((np.dot(X_train, c) - y_train[ind]) ** 2))
    print(np.sum((np.dot(X_test, c) - y_test[ind]) ** 2))

    pl.figure()
    pl.subplot(211)
    pl.plot(np.arange(len(y_train[ind])), y_train[ind], ".k")

    pl.subplot(212)
    pl.plot(np.arange(len(y_train[ind])), y_train[ind] - np.dot(X_train, c),
            ".k")
    pl.savefig("train.png")

    pl.figure()
    pl.subplot(211)
    pl.plot(np.arange(len(y_test[ind])), y_test[ind], ".k")

    pl.subplot(212)
    pl.plot(np.arange(len(y_test[ind])), y_test[ind] - np.dot(X_test, c),
            ".k")
    # pl.ylim(-0.001, 0.001)
    pl.savefig("test.png")
