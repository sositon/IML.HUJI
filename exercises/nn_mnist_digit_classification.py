import time
import numpy as np
import gzip
from typing import Tuple

from IMLearn.metrics.loss_functions import accuracy
from IMLearn.learners.neural_networks.modules import FullyConnectedLayer, ReLU, CrossEntropyLoss, softmax
from IMLearn.learners.neural_networks.neural_network import NeuralNetwork
from IMLearn.desent_methods import GradientDescent, StochasticGradientDescent, FixedLR
from IMLearn.utils.utils import confusion_matrix

import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt
pio.templates.default = "simple_white"


def load_mnist() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads the MNIST dataset

    Returns:
    --------
    train_X : ndarray of shape (60,000, 784)
        Design matrix of train set

    train_y : ndarray of shape (60,000,)
        Responses of training samples

    test_X : ndarray of shape (10,000, 784)
        Design matrix of test set

    test_y : ndarray of shape (10,000, )
        Responses of test samples
    """

    def load_images(path):
        with gzip.open(path) as f:
            # First 16 bytes are magic_number, n_imgs, n_rows, n_cols
            raw_data = np.frombuffer(f.read(), 'B', offset=16)
        # converting raw data to images (flattening 28x28 to 784 vector)
        return raw_data.reshape(-1, 784).astype('float32') / 255

    def load_labels(path):
        with gzip.open(path) as f:
            # First 8 bytes are magic_number, n_labels
            return np.frombuffer(f.read(), 'B', offset=8)

    return (load_images('../datasets/mnist-train-images.gz'),
            load_labels('../datasets/mnist-train-labels.gz'),
            load_images('../datasets/mnist-test-images.gz'),
            load_labels('../datasets/mnist-test-labels.gz'))


def plot_images_grid(images: np.ndarray, title: str = ""):
    """
    Plot a grid of images

    Parameters
    ----------
    images : ndarray of shape (n_images, 784)
        List of images to print in grid

    title : str, default="
        Title to add to figure

    Returns
    -------
    fig : plotly figure with grid of given images in gray scale
    """
    side = int(len(images) ** 0.5)
    subset_images = images.reshape(-1, 28, 28)

    height, width = subset_images.shape[1:]
    grid = subset_images.reshape(side, side, height, width).swapaxes(1, 2).reshape(height * side, width * side)

    return px.imshow(grid, color_continuous_scale="gray") \
        .update_layout(title=dict(text=title, y=0.97, x=0.5, xanchor="center", yanchor="top"),
                       font=dict(size=16), coloraxis_showscale=False) \
        .update_xaxes(showticklabels=False) \
        .update_yaxes(showticklabels=False)


if __name__ == '__main__':
    train_X, train_y, test_X, test_y = load_mnist()
    (n_samples, n_features), n_classes = train_X.shape, 10

    # ---------------------------------------------------------------------------------------------#
    # Question 5+6+7: Network with ReLU activations using SGD + recording convergence              #
    # ---------------------------------------------------------------------------------------------#
    # Initialize, fit and test network

    convergence_values = []
    grad_norm = []


    def convergence_callback(solver, weights, val, grad, t, eta, delta, batch_indices):
        convergence_values.append(val)
        grad_norm.append(np.linalg.norm(grad))
        return


    first_layer = FullyConnectedLayer(input_dim=n_features, output_dim=64, activation=ReLU())
    second_layer = FullyConnectedLayer(input_dim=64, output_dim=64, activation=ReLU())
    third_layer = FullyConnectedLayer(input_dim=64, output_dim=n_classes, activation=None, include_intercept=True)
    loss_func = CrossEntropyLoss()
    network = NeuralNetwork([first_layer, second_layer, third_layer], loss_func,
                            solver=StochasticGradientDescent(learning_rate=FixedLR(0.1), max_iter=10000, batch_size=256,
                                                             callback=convergence_callback))
    network.fit(train_X, train_y)
    print("accuracy for the test set = ", accuracy(test_y, network.predict(test_X)))

    # question 6
    # Plotting convergence process
    fig1 = go.Figure()

    fig1.add_trace(go.Scatter(x=np.arange(len(convergence_values)), y=convergence_values, mode='markers+lines',
                              name="loss values"))

    fig1.add_trace(go.Scatter(x=np.arange(len(convergence_values)), y=grad_norm, mode='markers+lines',
                              name="gradiant norm values"))

    fig1.update_layout(title="loss values and gradiant norm as a function of iteration number",
                       xaxis_title="number of iteration",
                       yaxis_title="loss values")

    fig1.show()

    # Question 7
    # Create a confusion matrix
    cm = confusion_matrix(test_y, network.predict(test_X))
    plt.imshow(confusion_matrix(test_y, network.predict(test_X)))
    plt.colorbar()
    plt.show()

    # ---------------------------------------------------------------------------------------------#
    # Question 8: Network without hidden layers using SGD                                          #
    # ---------------------------------------------------------------------------------------------#
    loss_func = CrossEntropyLoss()
    third_layer = FullyConnectedLayer(input_dim=n_features, output_dim=n_classes, activation=None, include_intercept=True)
    network1 = NeuralNetwork([third_layer], loss_func,
                            solver=StochasticGradientDescent(learning_rate=FixedLR(0.1), max_iter=10000,
                                                             batch_size=256))
    network1.fit(train_X, train_y)
    print("accuracy for the test set = ", accuracy(test_y, network1.predict(test_X)))

    # ---------------------------------------------------------------------------------------------#
    # Question 9: Most/Least confident predictions                                                 #
    # ---------------------------------------------------------------------------------------------#

    seven_test_X = test_X[test_y == 7]
    seven_test_y = test_y[test_y == 7]
    seven_confidence = [x[7] for x in network.compute_prediction(seven_test_X)]
    max_idxes = np.array(seven_confidence).argsort()[::-1][:64]
    min_idxes = np.array(seven_confidence).argsort()[:64]
    plot_images_grid(np.array(seven_test_X[max_idxes]),
                     "Most confident prediction (7)").show()
    plot_images_grid(np.array(seven_test_X[min_idxes]),
                     "Least confident prediction (7)").show()
    # ---------------------------------------------------------------------------------------------#
    # Question 10: GD vs GDS Running times                                                         #
    # ---------------------------------------------------------------------------------------------#
    loss_values = []
    time_values = []

    def time_callback(solver, weights, val, grad, t, eta, delta, batch_indices):
        loss_values.append(val)
        time_values.append(time.time())
        return

    first_layer = FullyConnectedLayer(input_dim=train_X.shape[1], output_dim=64, activation=ReLU())
    second_layer = FullyConnectedLayer(input_dim=64, output_dim=64, activation=ReLU())
    loss_func = CrossEntropyLoss()
    network_gd = NeuralNetwork([first_layer, second_layer], loss_func,
                               solver=GradientDescent(learning_rate=FixedLR(0.1), tol=1e-10, max_iter=10000,
                                                      callback=time_callback))
    network_sgd = NeuralNetwork([first_layer, second_layer], loss_func,
                                solver=StochasticGradientDescent(
                                    learning_rate=FixedLR(0.1), tol=1e-10, max_iter=10000,
                                    batch_size=64, callback=time_callback))

    # Fit and plot GD network
    gd_start_time = time.time()
    network_gd.fit(train_X[np.arange(2500)], train_y[np.arange(2500)])
    lists_len = len(loss_values)
    sgd_start_time = time.time()
    network_sgd.fit(train_X[np.arange(2500)], train_y[np.arange(2500)])

    gd_loss_values = loss_values[:lists_len]
    gd_time_values = [x - gd_start_time for x in time_values[:lists_len]]
    sgd_loss_values = loss_values[lists_len:]
    sgd_time_values = [x - sgd_start_time for x in time_values[lists_len:]]

    # GD graph showing the running time vs. loss
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=gd_time_values, y=gd_loss_values))
    fig1.update_layout(title="GD running time vs loss",
                       xaxis_title="running time",
                       yaxis_title="loss values")
    fig1.show()

    # SGD graph showing the running time vs. loss
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=sgd_time_values, y=sgd_loss_values))
    fig2.update_layout(title="SGD running time vs loss",
                       xaxis_title="running time",
                       yaxis_title="loss values")
    fig2.show()

    # Graph of both solvers one on top of the other as two different scatters
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=gd_time_values, y=gd_loss_values, name="GD"))
    fig3.add_trace(go.Scatter(x=sgd_time_values, y=sgd_loss_values, name="SGD"))
    fig3.update_layout(title="Both solvers running time vs loss",
                       xaxis_title="running time",
                       yaxis_title="loss values")
    fig3.show()

