from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, \
    RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions

    # generate the data samples
    X = np.linspace(-1.2, 2, n_samples)
    e = np.random.normal(0, noise, n_samples)
    f = lambda x: (x + 3)*(x + 2)*(x + 1)*(x - 1)*(x - 2)
    y = f(X) + e
    train_portion = 2/3
    X_train, y_train, X_test, y_test = split_train_test(pd.DataFrame(X), pd.Series(y), train_portion)
    fig = go.Figure()
    # Add traces
    fig.add_trace(go.Scatter(x=X, y=f(X),
                             mode='lines+ markers',
                             name='Noiseless Model',
                             showlegend=True))
    fig.add_trace(go.Scatter(x=np.array(X_train).reshape(-1), y=np.array(y_train),
                             mode='markers',
                             name='Train',
                             showlegend=True))
    fig.add_trace(go.Scatter(x=np.array(X_test).reshape(-1), y=np.array(y_test),
                             mode='markers',
                             name='Test',
                             showlegend=True))
    fig.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    max_degree = 10 + 1  # include ten
    k_train_scores, k_valid_scores = [], []
    for k in range(max_degree):
        ts, vs = cross_validate(PolynomialFitting(k), np.array(X_train).reshape(-1), np.array(y_train).reshape(-1), mean_square_error)
        k_train_scores.append(ts)
        k_valid_scores.append(vs)

    fig = go.Figure()
    # Add traces
    fig.add_trace(go.Scatter(x=list(range(max_degree)), y=k_train_scores,
                             mode='lines',
                             name='Train Scores',
                             showlegend=True))
    fig.add_trace(go.Scatter(x=list(range(max_degree)), y=k_valid_scores,
                             mode='lines',
                             name='Validation Scores',
                             showlegend=True))
    fig.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error


def select_regularization_parameter(n_samples: int = 50,
                                    n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    raise NotImplementedError()

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    raise NotImplementedError()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
