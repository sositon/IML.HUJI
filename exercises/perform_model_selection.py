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
    X = np.sort(np.random.uniform(-1.2, 2, n_samples))
    e = np.random.normal(0, noise, n_samples)
    f = lambda x: (x + 3)*(x + 2)*(x + 1)*(x - 1)*(x - 2)
    y = f(X) + e
    train_portion = 2/3
    X_train, y_train, X_test, y_test = split_train_test(pd.DataFrame(X), pd.Series(y), train_portion)
    X_train, y_train, X_test, y_test = np.array(X_train).reshape(-1), np.array(y_train), np.array(X_test).reshape(-1), np.array(y_test)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X, y=f(X),
                             mode='lines+ markers',
                             name='Noiseless Model',
                             showlegend=True))
    fig.add_trace(go.Scatter(x=X_train, y=y_train,
                             mode='markers',
                             name='Train',
                             showlegend=True))
    fig.add_trace(go.Scatter(x=X_test, y=y_test,
                             mode='markers',
                             name='Test',
                             showlegend=True))
    fig.update_layout(title="The True Model and the Two Sets of Train Test",
                      xaxis={"title": "X"},
                      yaxis={"title": "F(x)"})
    fig.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    max_degree = 10 + 1  # include ten
    k_train_scores, k_valid_scores = [], []
    for k in range(max_degree):
        ts, vs = cross_validate(PolynomialFitting(k), X_train, y_train, mean_square_error)
        k_train_scores.append(ts)
        k_valid_scores.append(vs)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(max_degree)), y=k_train_scores,
                             mode='lines',
                             name='Train Scores',
                             showlegend=True))
    fig.add_trace(go.Scatter(x=list(range(max_degree)), y=k_valid_scores,
                             mode='lines',
                             name='Validation Scores',
                             showlegend=True))
    fig.update_layout(title="Training and Validation Errors as function of K",
                      xaxis={"title": "K"},
                      yaxis={"title": "MSE"})
    fig.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    best_k = np.argmin(k_valid_scores)
    kpm = PolynomialFitting(k=best_k).fit(X_train, y_train)
    print(round(mean_square_error(kpm.predict(X_test), y_test), 2))


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
    X, y = datasets.load_diabetes(return_X_y=True)
    X_train, y_train = X[:n_samples, :], y[:n_samples]
    X_test, y_test = X[n_samples:, :], y[n_samples:]
    lambdas = np.linspace(0.01, 2, n_evaluations)

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    ridge_train_scores, ridge_valid_scores = [], []
    lasso_train_scores, lasso_valid_scores = [], []
    for lam in lambdas:
        ts, vs = cross_validate(RidgeRegression(lam), X_train, y_train, mean_square_error)
        ridge_train_scores.append(ts)
        ridge_valid_scores.append(vs)
        ts, vs = cross_validate(Lasso(alpha=lam), X_train, y_train, mean_square_error)
        lasso_train_scores.append(ts)
        lasso_valid_scores.append(vs)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lambdas, y=ridge_train_scores,
                             mode='lines',
                             name='Ridge Train Scores',
                             showlegend=True))
    fig.add_trace(go.Scatter(x=lambdas, y=ridge_valid_scores,
                             mode='lines',
                             name='Ridge Validation Scores',
                             showlegend=True))
    fig.add_trace(go.Scatter(x=lambdas, y=lasso_train_scores,
                             mode='lines',
                             name='Lasso Train Scores',
                             showlegend=True))
    fig.add_trace(go.Scatter(x=lambdas, y=lasso_valid_scores,
                             mode='lines',
                             name='Lasso Validation Scores',
                             showlegend=True))
    fig.update_layout(title="Training and Validation Errors as function of Lambda",
                      xaxis={"title": "Lambda"},
                      yaxis={"title": "MSE"})
    fig.show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    ridge_best_lam = lambdas[np.argmin(ridge_valid_scores)]
    lasso_best_lam = lambdas[np.argmin(lasso_valid_scores)]
    print("Lambda for Ridge: ", ridge_best_lam)
    print("Lambda for Lasso: ", lasso_best_lam)
    r = RidgeRegression(ridge_best_lam).fit(X_train, y_train)
    l = Lasso(lasso_best_lam).fit(X_train, y_train)
    lr = LinearRegression().fit(X_train, y_train)
    print("Error for LinearRegression: ",
          mean_square_error(y_test, lr.predict(X_test)))
    print("Error for Ridge: ",
          mean_square_error(y_test, r.predict(X_test)))
    print("Error for Lasso: ",
          mean_square_error(y_test, l.predict(X_test)))


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    # Question 4
    select_polynomial_degree(noise=0)
    # Question 5
    select_polynomial_degree(n_samples=1500, noise=10)
    select_regularization_parameter()
