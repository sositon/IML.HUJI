from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """
    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_ = np.unique(y)
        K = len(self.classes_)
        m, d = X.shape
        nk = {k: np.count_nonzero(y == k) for k in self.classes_}
        self.mu_ = np.array([1/nk[k] * np.sum(X[i] if y[i] == k else 0 for i in range(m)) for k in nk])
        self.pi_ = np.array([nk[k] / m for k in nk])
        cov_k = np.array([np.sum(np.transpose((X[i] - self.mu_[k]).reshape(1, -1)) @ (X[i] - self.mu_[k]).reshape(1, -1)
                                 if y[i] == k else 0 for i in range(m)) for k in nk])
        self.cov_ = np.zeros(shape=(d, d))
        for k, cov in enumerate(cov_k):
            self.cov_ += cov
        self.cov_ /= m - K
        self._cov_inv = inv(self.cov_)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        ak = np.array([self._cov_inv @ np.transpose(self.mu_[k]) for k in self.classes_])
        bk = np.array([np.log(self.pi_[k]) - 0.5 * self.mu_[k] @ ak[k] for k in self.classes_])
        y_hat = ak @ X.T + bk.reshape(-1, 1)
        return np.argmax(y_hat, axis=0)

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")
        from scipy.stats import multivariate_normal as mn
        res = np.zeros(shape=(X.shape[0], len(self.classes_)))
        for k in range(len(self.classes_)):
            # We calculate Log Likelihood
            res[:, k] = mn(self.mu_[k], self.cov_).logpdf(X) + np.log(self.pi_[k])
        return res

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        return misclassification_error(y, self._predict(X))
