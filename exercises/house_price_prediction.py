from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def cov(x, y):
    return np.sum((x - np.mean(x)) * (y - np.mean(y))) / (len(y) - 1)


def filter1(data: pd.DataFrame) -> pd.DataFrame:
    """
    drops NaN values
    """
    # check how many null values there are
    # not many rows are having nan values so we'll drop those
    data.dropna(inplace=True)
    return data


def filter2(data: pd.DataFrame) -> pd.DataFrame:
    """
    drops values bellow 0 that doesn't have many occurrences
    with other values that does have many occurrences we'll leave to the next filter
    """
    # check how many values that are equal to zero there are
    # print(data[data == 0].count())
    # filter zero values with few zero occurrences
    data = data.loc[data["price"] > 0]
    data = data.loc[data["bedrooms"] > 0]
    data = data.loc[data["bathrooms"] > 0]
    # print(data[data == 0].count())
    # great improvement !
    return data


def filter3(data: pd.DataFrame) -> pd.DataFrame:
    """
    dealing with outliers
    """
    del data["id"]
    # gave up on that categorical feature

    # data["date"] = data["date"].apply(lambda x: x[:6])
    # date_dummies = pd.get_dummies(data["date"], prefix="date", prefix_sep="_")
    # print(data.date.value_counts().sort_index())
    del data["date"]

    # turn type to int int
    # data.bathrooms = data.bathrooms.astype(int)
    # data.view = data.view.astype(int)
    # data.condition = data.condition.astype(int)
    # data.grade = data.grade.astype(int)
    # data.sqft_basement = data.sqft_basement.astype(int)

    # turn yr_renovated to binary
    data.yr_renovated = np.where(data.yr_renovated == 0, 0, 1)
    # turn zip to categorical feature
    zip_dummies = pd.get_dummies(data["zipcode"], prefix="zip", prefix_sep="_")
    del data["zipcode"]
    # derive living and lot 'effect'
    data["living_effect"] = data.apply(
        lambda row: np.sign(row["sqft_living"] - row["sqft_living15"]), axis=1)
    data["lot_effect"] = data.apply(
        lambda row: np.sign(row["sqft_lot"] - row["sqft_lot15"]), axis=1)
    del data["sqft_living15"]
    del data["sqft_lot15"]

    return pd.concat([data, zip_dummies], axis=1)


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    # read
    raw_data = pd.read_csv(filename)
    filtered_data_1 = filter1(raw_data)
    filtered_data_2 = filter2(filtered_data_1)
    filtered_data_3 = filter3(filtered_data_2)
    filtered_data_4 = filter1(filtered_data_3)
    price = filtered_data_4.pop("price")

    return filtered_data_4, price


def feature_evaluation(X: pd.DataFrame, y: pd.Series,
                       output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    std_err_y = np.std(y)
    for f in X:
        feature = X[f]
        std_err_mul = np.std(feature) * std_err_y
        corr = cov(feature, y) / std_err_mul
        ratio = y.max() / feature.max() if feature.max() != 0 else 10 ** 6
        go.Figure(
            [go.Scatter(x=feature, y=y, mode="markers", line=dict(width=4),
                        name='r$Feature,Response$', showlegend=True),
             go.Scatter(x=feature, y=ratio * feature * corr, mode="lines",
                        line=dict(width=5, color="rgb(204,68,83)"),
                        name=f"r$Corr = {corr.round(3)}$", showlegend=True)],
            layout=go.Layout(barmode='overlay',
                             title=r"$\text{Feature Correlation}$",
                             xaxis_title=f"{f}",
                             yaxis_title="r$Prices$")).write_image(fr"{output_path}feature_{f}.png")


def fit_model_over_increase_samples(train_X, train_y, test_X, test_y):
    """
    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    """
    lr = LinearRegression()
    pp = np.linspace(10, 100, 91)
    loss_all_samples = []
    loss_all_samples_plus = []
    loss_all_samples_minus = []
    for p in range(10, 101):
        loss_ = []
        for i in range(10):
            train_p_X = train_X.sample(frac=p / 100)
            train_p_y = train_y.reindex(train_p_X.index)
            lr.fit(train_p_X, train_p_y)
            loss_.append(lr.loss(test_X, test_y))
        curr_mean = np.mean(loss_)
        cur_std = np.std(loss_)
        loss_all_samples.append(curr_mean)
        loss_all_samples_minus.append(curr_mean - 2 * cur_std)
        loss_all_samples_plus.append(curr_mean + 2 * cur_std)

    go.Figure(
        [go.Scatter(x=pp, y=loss_all_samples, mode="lines", line=dict(width=4),
                    name='r$MSE$', showlegend=True),
         go.Scatter(x=pp, y=loss_all_samples_plus, mode="lines",
                    line=dict(width=4),
                    name='r$MSE + 2 std$', showlegend=True),
         go.Scatter(x=pp, y=loss_all_samples_minus, mode="lines",
                    line=dict(width=4),
                    name='r$MSE - 2 std$', showlegend=True)],
        layout=go.Layout(barmode='overlay',
                         title=r"$\text{Average Loss as Function of Training Size With Error Ribbon of Size}$",
                         xaxis_title=f"$Prcentage$",
                         yaxis_title="r$MSE$")).show()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    matrix, response = load_data(
        "/Users/omersiton/IML.HUJI/datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    # feature_evaluation(matrix, response, "/Users/omersiton/IML.HUJI/exercises/features_photos/")

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(matrix, response)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    fit_model_over_increase_samples(train_X, train_y, test_X, test_y)
