import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
from typing import NoReturn
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go

pio.templates.default = "simple_white"


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
    # check values distribution
    # print(data["Country"].value_counts().sort_index())
    # print(data["City"].value_counts().sort_index())
    # print(data["Temp"].value_counts().sort_index())
    # filter temp == -72 values, consider as noise
    data = data.loc[data["Temp"] > -70]
    # great improvement !
    return data


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    raw_data = pd.read_csv(filename, parse_dates=["Date"])
    # null and noisy values filter
    filtered_data_1 = filter1(raw_data)
    filtered_data_2 = filter2(filtered_data_1)
    filtered_data_2["DayOfYear"] = filtered_data_2.Date.apply(lambda row: row.timetuple().tm_yday)

    return filtered_data_2


def question_2(df: pd.DataFrame) ->NoReturn:
    isr_data = df.loc[df["Country"] == "Israel"]
    years_ = set(df["Year"])
    data_ = []
    for yr in years_:
        temp_data = isr_data.loc[df["Year"] == yr]
        data_.append(go.Scatter(x=temp_data["DayOfYear"], y=temp_data["Temp"], mode="markers",
                                name=f"r${yr}$", showlegend=True))

    go.Figure(data_, layout=go.Layout(barmode='overlay',
                                           title=r"$Temperature In Israel Throughout The Years$",
                                           xaxis_title=f"$Day Of Year$",
                                           yaxis_title="r$Temperature$")).show()

    std = round(isr_data.groupby(["Month"])["Temp"].agg('std'), 2)
    months_ = list(range(1, 13))
    px.bar(x=months_, y=std, title="r$Israel Months STD$", text=std).show()


def question_3(df: pd.DataFrame) -> NoReturn:
    std_data = df.groupby(["Country", "Month"])["Temp"].agg('std')
    average_data = df.groupby(["Country", "Month"])["Temp"].agg('mean')
    countries_ = set(df["Country"])
    months_ = list(range(1, 13))
    data_ = []
    for c in countries_:
        data_.append(go.Scatter(x=months_, y=average_data[c],
                                error_y=dict(type='data',
                                             array=std_data[c],
                                             visible=True),
                                mode="markers+lines",
                                name=f"r${c}$", showlegend=True))
    go.Figure(data_, layout=go.Layout(barmode='overlay',
                                      title=r"$Average Temperature Throughout The Years$",
                                      xaxis_title=f"$Day Of Year$",
                                      yaxis_title="r$Average Temperature$")).show()


def question_4(df: pd.DataFrame) -> NoReturn:
    isr_data = df.loc[df["Country"] == "Israel"]
    train_X, train_y, test_X, test_y = split_train_test(isr_data["DayOfYear"], isr_data["Temp"])
    loss_ = []
    for k in range(1, 11):
        pf = PolynomialFitting(k)
        pf.fit(np.array(train_X), np.array(train_y))
        loss_.append(round(pf.loss(np.array(test_X), np.array(test_y)), 2))

    print(loss_)
    px.bar(x=list(range(1, 11)), y=loss_, title="r$Loss Values by K$", text=loss_).show()

def question_5(df: pd.DataFrame) -> NoReturn:
    isr_data = df[df["Country"] == "Israel"]
    countries_ = ["Jordan", "The Netherlands", "South Africa"]
    pf = PolynomialFitting(5)
    train_X, train_y = np.array(isr_data["DayOfYear"]), np.array(isr_data["Temp"])
    pf.fit(train_X, train_y)
    loss_ = []
    for c in countries_:
        temp_data = df[df["Country"] == c]
        test_X, test_y = np.array(temp_data["DayOfYear"]), np.array(temp_data["Temp"])
        loss_.append(round(pf.loss(test_X, test_y), 2))
    px.bar(x=countries_, y=loss_, title="r$Loss Values by K$", text=loss_).show()

if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    matrix = load_data(
        "/Users/omersiton/IML.HUJI/datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    # isr_data = matrix.loc[matrix["Country"] == "Israel"]
    question_2(matrix)
    # Question 3 - Exploring differences between countries
    question_3(matrix)

    # Question 4 - Fitting model for different values of `k`
    question_4(matrix)

    # Question 5 - Evaluating fitted model on different countries
    question_5(matrix)
