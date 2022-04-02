from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def remove_outliers_IQR(data: pd.DataFrame, feature: str) -> pd.DataFrame:
    Q75 = np.percentile(data['price'], 75)
    Q25 = np.percentile(data['price'], 25)
    IQR = Q75 - Q25
    cutoff = IQR * 1.5
    upper = Q75 + cutoff

    return data[(data['price'] < upper)]


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

    data["date"] = data["date"].apply(lambda x: x[:6])
    date_dummies = pd.get_dummies(data["date"], prefix="date", prefix_sep="_")
    # print(data.date.value_counts().sort_index())
    del data["date"]

    data.bathrooms = data.bathrooms.astype(int)
    data = data.loc[data["bathrooms"] < 7]

    data = data.loc[data["bedrooms"] < 9]

    data = data.loc[data["floors"] < 3.5]

    data.view = data.view.astype(int)

    data.condition = data.condition.astype(int)

    data.grade = data.grade.astype(int)

    data.sqft_basement = data.sqft_basement / 100
    data.sqft_basement = data.sqft_basement.astype(int)

    data.yr_renovated = np.where(data.yr_renovated == 0, 0, 1)

    data.zipcode = (data.zipcode % 200).astype(int)
    zip_dummies = pd.get_dummies(data["zipcode"], prefix="zip", prefix_sep="_")
    del data["zipcode"]
    del data["lat"]
    del data["long"]

    data["living_effect"] = data.apply(lambda row: np.sign(row["sqft_living"] - row["sqft_living15"]), axis=1)
    data["lot_effect"] = data.apply(lambda row: np.sign(row["sqft_lot"] - row["sqft_lot15"]), axis=1)
    del data["sqft_living15"]
    del data["sqft_lot15"]
    # print(data.living_effect.value_counts().sort_index())
    # print(data.lot_effect.value_counts().sort_index())
    # print(data)

    return pd.concat([data, date_dummies, zip_dummies], axis=1)


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
    # first filter all invalid values
    filtered_data_1 = filter1(raw_data)
    filtered_data_2 = filter2(filtered_data_1)
    filtered_data_3 = filter3(filtered_data_2)
    price = filtered_data_3.pop("price")
    return filtered_data_3, price


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
    figure = go.Figure([go.Scatter(x=np.sort(X), y=empiricalPDF, mode="lines+markers", line=dict(width=4),
                   name=r'$N(\mu, \frac{\sigma^2}{m1})$')],
                    layout=go.Layout(barmode='overlay',
                         title=r"$\text{Empirical PDF}$",
                         xaxis_title="r$Value$",
                         yaxis_title="r$Density$",
                         height=300))


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    matrix, response = load_data("/Users/omersiton/IML.HUJI/datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    # raise NotImplementedError()

    # Question 3 - Split samples into training- and testing sets.
    # raise NotImplementedError()

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    # raise NotImplementedError()
