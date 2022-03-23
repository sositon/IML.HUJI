from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mu = 10
    sigma = 1
    m = 1000
    X = np.random.normal(mu, sigma, m)
    u = UnivariateGaussian()
    u.fit(X)
    print(u.mu_, u.var_)

    # Question 2 - Empirically showing sample mean is consistent
    ms = np.linspace(10, 1000, 100).astype(np.int_)
    d = []
    for i in ms:
        d.append(abs(u.fit(X[:i]).mu_ - mu))
    go.Figure(
        [go.Scatter(x=ms, y=d, mode='markers+lines', name=r'$\widehat\mu$')],
        layout=go.Layout(title=r"$\text{Estimation of "
                               r"Distance Expectation As "
                               r"Function Of Number Of Samples}$",
                         xaxis_title="$m\\text{ - number of samples}$",
                         yaxis_title="r$|\hat\mu - \mu|$",
                         height=300)).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    empiricalPDF = u.pdf(np.sort(X))
    go.Figure([
        go.Scatter(x=np.sort(X), y=empiricalPDF, mode="lines+markers", line=dict(width=4),
                   name=r'$N(\mu, \frac{\sigma^2}{m1})$')],
        layout=go.Layout(barmode='overlay',
                         title=r"$\text{Empirical PDF}$",
                         xaxis_title="r$Value$",
                         yaxis_title="r$Density$",
                         height=300)).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    m = MultivariateGaussian()
    mu = np.array([0, 0, 4, 0])
    sigma = np.array([[1, 0.2, 0, 0.5],
                      [0.2, 2, 0, 0],
                      [0, 0, 1, 0],
                      [0.5, 0, 0, 1]])
    X = np.random.multivariate_normal(mu, sigma, 1000)
    m.fit(X)
    print(m.mu_, "\n", m.cov_)
    # Question 5 - Likelihood evaluation
    f1 = np.linspace(-10, 10, 200)
    f3 = np.linspace(-10, 10, 200)
    d = []
    for i in range(200):
        d.append([m.log_likelihood(np.array([f1[i], 0, f3[j], 0]), sigma, X) for j in range(200)])
    go.Figure(data=go.Heatmap(x=f3, y=f1, z=d, name="r$Log-Likelihood$"),
              layout=go.Layout(title="r$Heatmap - LogLikelihood$", height=600
                               , xaxis_title="r$F3 - Values$", yaxis_title="r$F1 - Values$")).show()

    # Question 6 - Maximum likelihood
    max_f1, max_f3 = np.unravel_index(np.argmax(d), np.array(d).shape)
    print(f1[max_f1].round(3), f3[max_f3].round(3))



if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
