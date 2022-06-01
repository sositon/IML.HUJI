import numpy as np
from ..base import BaseEstimator
from typing import Callable, NoReturn


class AdaBoost(BaseEstimator):
    """
    AdaBoost class for boosting a specified weak learner

    Attributes
    ----------
    self.wl_: Callable[[], BaseEstimator]
        Callable for obtaining an instance of type BaseEstimator

    self.iterations_: int
        Number of boosting iterations to perform

    self.models_: List[BaseEstimator]
        List of fitted estimators, fitted along the boosting iterations
    """

    def __init__(self, wl: Callable[[], BaseEstimator], iterations: int):
        """
        Instantiate an AdaBoost class over the specified base estimator

        Parameters
        ----------
        wl: Callable[[], BaseEstimator]
            Callable for obtaining an instance of type BaseEstimator

        iterations: int
            Number of boosting iterations to perform
        """
        super().__init__()
        self.wl_ = wl
        self.iterations_ = iterations
        self.models_, self.weights_, self.D_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an AdaBoost classifier over given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        m = X.shape[0]
        self.models_ = list()
        self.D_, self.weights_ = (np.ones(m) / m), np.zeros(self.iterations_)
        for t in range(self.iterations_):
            self.models_.append(self.wl_().fit(X, y*self.D_))
            y_hat = self.models_[t].predict(X)
            # indicator(y!=yhat) <-> |y-yhat|/2
            e = np.sum(self.D_ * (np.abs(y - y_hat) / 2))
            # weights
            self.weights_[t] = (1/2) * (np.log((1/e) - 1))
            # update sample weights and normalize
            self.D_ = self.D_ * np.exp((-y)*(self.weights_[t]*y_hat))
            self.D_ = self.D_ / np.sum(self.D_)

    def _predict(self, X: np.ndarray, iterations: int = None) -> NoReturn:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        iterations : int if None, full predict else
            The number of classifiers to be used for prediction

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        iterations = self.iterations_ if iterations is None else iterations
        m = X.shape[0]
        y_hat = np.zeros(m)
        for t in range(iterations):
            y_hat += (self.weights_[t] * self.models_[t].predict(X))
        return np.sign(y_hat)

    def _loss(self, X: np.ndarray, y: np.ndarray, iterations: int = None) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        iterations : int if None, full predict else
            The number of classifiers to be used for prediction

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        iterations = self.iterations_ if iterations is None else iterations
        from ..metrics.loss_functions import misclassification_error
        return misclassification_error(y, self._predict(X, iterations))

    def partial_predict(self, X: np.ndarray, T: int) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimators

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        return self._predict(X, T)

    def partial_loss(self, X: np.ndarray, y: np.ndarray, T: int) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return self._loss(X, y, T)
