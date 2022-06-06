from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def flatten(arr):
    # list of ndarrays
    res = []
    for ls in arr:
        for val in ls:
            res.append(val)
    return np.array(res)


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    # split X to cv number of folds and the same for y respectively
    X_split = np.array_split(X, cv)
    y_split = np.array_split(y, cv)
    train_scores, validation_scores = [], []
    for i, X_valid in enumerate(X_split):
        y_valid = y_split[i]
        X_train = flatten(X_split[:i] + X_split[i + 1:])
        y_train = flatten(y_split[:i] + y_split[i + 1:])
        # fit on the K-1 folds
        estimator.fit(X_train, y_train)
        train_scores.append(scoring(estimator.predict(X_train), y_train))
        validation_scores.append(scoring(estimator.predict(X_valid), y_valid))

    return np.mean(train_scores), np.mean(validation_scores)


