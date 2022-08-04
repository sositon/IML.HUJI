from __future__ import annotations
from typing import Callable, Tuple
import numpy as np

from IMLearn.base import BaseModule, BaseLR
from IMLearn.desent_methods.gradient_descent import default_callback
from IMLearn.desent_methods.learning_rate import FixedLR


class StochasticGradientDescent:
    """
    Gradient Descent algorithm

    Attributes:
    -----------
    learning_rate_: BaseLR
        Learning rate strategy for retrieving the learning rate at each iteration t of the algorithm

    tol_: float
        The stopping criterion. Training stops when the Euclidean norm of w^(t)-w^(t-1) is less than
        specified tolerance

    max_iter_: int
        The maximum number of GD iterations to be performed before stopping training

    callback_: Callable[[...], None], default=default_callback
        A callable function to be called after each update of the model while fitting to given data.
        Callable function receives as input any argument relevant for the current GD iteration. Arguments
        are specified in the `GradientDescent.fit` function
    """

    def __init__(self,
                 learning_rate: BaseLR = FixedLR(1e-3),
                 tol: float = 1e-5,
                 max_iter: int = 1000,
                 batch_size: int = 1,
                 callback: Callable[[...], None] = default_callback):
        """
        Instantiate a new instance of the GradientDescent class

        Parameters
        ----------
        learning_rate: BaseLR, default=FixedLR(1e-3)
            Learning rate strategy for retrieving the learning rate at each iteration t of the algorithm

        tol: float, default=1e-5
            The stopping criterion. Training stops when the Euclidean norm of w^(t)-w^(t-1) is less than
            specified tolerance

        max_iter: int, default=1000
            The maximum number of GD iterations to be performed before stopping training

        batch_size: int, default=1
            Number of samples to randomly select at each iteration of the SGD algorithm

        callback: Callable[[...], None], default=default_callback
            A callable function to be called after each update of the model while fitting to given data.
            Callable function receives as input any argument relevant for the current GD iteration. Arguments
            are specified in the `GradientDescent.fit` function
        """
        self._learning_rate = learning_rate
        self._tol = tol
        self._max_iter = max_iter
        self._callback = callback
        self._batch_size = batch_size

    def fit(self, f: BaseModule, X: np.ndarray, y: np.ndarray):
        """
        Optimize module using SGD iterations over given input samples and responses

        Parameters
        ----------
        f : BaseModule
        Objective function (module) to be minimized by SGD


        X : ndarray of shape (n_samples, n_features)
            Input data to optimize module over

        y : ndarray of shape (n_samples, )
            Responses of input data to optimize module over

        Returns
        -------
        solution: ndarray of shape (n_features)
            Obtained solution for module optimization

        Notes
        -----
        - Optimization is performed as long as self.max_iter_ has not been reached and that
        Euclidean norm of w^(t)-w^(t-1) is more than the specified self.tol_

        - At each iteration the learning rate is specified according to self.learning_rate_.lr_step

        - At the end of each iteration the self.callback_ function is called passing self and the
        following named arguments:
            - solver: GradientDescent
                self, the current instance of GradientDescent
            - weights: ndarray of shape specified by module's weights
                Current weights of objective
            - val: ndarray of shape specified by module's compute_output function
                Value of objective function at current point, over given data X, y
            - grad:  ndarray of shape specified by module's compute_jacobian function
                Module's jacobian with respect to the weights and at current point, over given data X, y
            - t: int
                Current GD iteration
            - eta: float
                Learning rate used at current iteration
            - delta: float
                Euclidean norm of w^(t)-w^(t-1)
            - batch_indices: np.ndarray of shape (n_batch,)
                Sample indices used in current SGD iteration
        """
        if f.weights is None:
            f.weights = np.random.normal(size=X.shape[1])
        w_sum = 0
        for t in range(self._max_iter):
            w_t = f.weights
            batch_ind = np.random.choice(len(X), self._batch_size, replace=False)
            val, grad, eta = self._partial_fit(f, X[batch_ind], y[batch_ind], t)
            w_t_1 = f.weights
            delta = np.linalg.norm(w_t_1 - w_t)
            self._callback(solver=self, weights=w_t_1, val=val, grad=grad, t=t,
                           eta=eta, delta=delta, batch_indices=batch_ind)
            w_sum += w_t_1
            if delta < self._tol:
                return w_sum / t
        return w_sum / self._max_iter

    def _partial_fit(self, f: BaseModule, X: np.ndarray, y: np.ndarray, t: int) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Perform a SGD iteration over given samples

        Parameters
        ----------
        f : BaseModule
        Objective function (module) to be minimized by SGD

        X : ndarray of shape (n_batch, n_features)
            Input data to optimize module over

        y : ndarray of shape (n_batch, )
            Responses of input data to optimize module over

        t: int
            Current SGD iteration

        Returns
        -------
        val: ndarray of shape (n_features,)
            Value of objective optimized, at current position, based on given samples

        jac: ndarray of shape (n_features, )
            Jacobian on objective optimized, at current position, based on given samples

        eta: float
            learning rate used at current iteration
        """
        w_t = f.weights
        grad = f.compute_jacobian(X, y)
        # eta = self._learning_rate.lr_step(t)
        eta = self._learning_rate.lr_step()
        w_t_1 = w_t - eta * grad
        f.weights = w_t_1
        val = f.compute_output(X, y)
        return val, f.compute_jacobian(X, y), eta


