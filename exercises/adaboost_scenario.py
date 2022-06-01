import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def q1_error_as_func_of_learners(train_err, test_err, n_learners):
    fig = go.Figure(
        [go.Scatter(x=np.linspace(1, n_learners, n_learners), y=train_err, name="Train Error",
                    showlegend=True, mode="lines"),
         go.Scatter(x=np.linspace(1, n_learners, n_learners), y=test_err,
                    name="Test Error",
                    showlegend=True, mode="lines")
         ],
        layout=go.Layout(title=r"$\text{(1) Errors as Function of Number of Learners}$",
                         xaxis={
                             "title": "Number of Learners"},
                         yaxis={"title": "MissClassification Error"}))
    fig.show()


def q2_boundaries_as_function_of_learners(ada, T, X, Y, lims):
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=[rf"$\textbf{{{m}}}$" for m in T],
                        horizontal_spacing=0.01, vertical_spacing=.03)
    for i, t in enumerate(T):
        fig.add_traces([decision_surface(lambda x: ada.partial_predict(x, t),
                                         lims[0], lims[1], showscale=False),
                        go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                                   showlegend=False,
                                   marker=dict(color=Y, symbol="circle",
                                               colorscale=[custom[0],
                                                           custom[-1]],
                                               line=dict(color="black",
                                                         width=1)))],
                       rows=(i // 2) + 1, cols=(i % 2) + 1)
    fig.update_layout(
        title=rf"$\textbf{{(2) Decision Boundaries Obtained by Different Iterations}}$",
        margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)
    fig.show()


def q3_best_preforming_ensemble(ada, X, Y, best_idx, lims, test_err):
    fig = make_subplots(rows=1, cols=1,
                        horizontal_spacing=0.01, vertical_spacing=.03)
    fig.add_traces(
        [decision_surface(lambda x: ada.partial_predict(x, best_idx),
                          lims[0], lims[1], showscale=False),
         go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                    showlegend=False,
                    marker=dict(color=Y, symbol="circle",
                                colorscale=[custom[0],
                                            custom[-1]],
                                line=dict(color="black",
                                          width=1)))],
        rows=1, cols=1)
    fig.update_layout(
        title=rf"$\textbf{{(3) Decision Surface of Best Performing Ensemble - T = {best_idx} , accuracy = {1 - test_err[best_idx - 1]}}}$",
        margin=dict(t=100)).update_xaxes(visible=False).update_yaxes(visible=False)
    fig.show()


def q4_(ada, X, Y, lims, weights):
    fig = make_subplots(rows=1, cols=1,
                        horizontal_spacing=0.01, vertical_spacing=.03)
    fig.add_traces(
        [decision_surface(ada.predict,
                          lims[0], lims[1], showscale=False),
         go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                    showlegend=False,
                    marker=dict(color=Y, symbol="circle", size=weights,
                                colorscale=[custom[0],
                                            custom[-1]],
                                line=dict(color="black",
                                          width=1)))],
        rows=1, cols=1)
    fig.update_layout(
        title=rf"$\textbf{{(4) Decision Surface of Fitted Ensemble with Weights}}$",
        margin=dict(t=100)).update_xaxes(visible=False).update_yaxes(visible=False)
    fig.show()


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    for n in noise:
        (train_X, train_y), (test_X, test_y) = generate_data(train_size, n), generate_data(test_size, n)

        # Question 1: Train- and test errors of AdaBoost in noiseless case
        ada = AdaBoost(DecisionStump, n_learners).fit(train_X, train_y)
        train_err = [ada.partial_loss(train_X, train_y, i) for i in range(1, n_learners + 1)]
        test_err = [ada.partial_loss(test_X, test_y, i) for i in range(1, n_learners + 1)]
        q1_error_as_func_of_learners(train_err, test_err, n_learners)

        # Question 2: Plotting decision surfaces
        T = [5, 50, 100, 250]
        X = np.r_[train_X, test_X]
        Y = np.r_[train_y, test_y]
        lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
        if (n == 0):
            q2_boundaries_as_function_of_learners(ada, T, X, Y, lims)

            # Question 3: Decision surface of best performing ensemble
            # todo check what loss function should return
            best_idx = np.argmin(test_err) + 1
            q3_best_preforming_ensemble(ada, X, Y, best_idx, lims, test_err)

        # Question 4: Decision surface with weighted samples
        magnifier = 7
        weights = ada.D_ / np.max(ada.D_) * magnifier
        # todo check if we should pass just the train set
        q4_(ada, train_X, train_y, lims, weights)


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost([0, 0.4])
