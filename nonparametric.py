import numpy as np
from scipy.stats import norm


class Regressogram:
    def __init__(self, m, a=0, b=1):
        h = (b - a) / m
        self.m = m
        self.bins = a + h * np.arange(m + 1)
        self.bins_means = np.nan * np.ones(m + 2)
        # Last (and only last) interval should be closed
        self.bins[-1] += np.finfo(float).eps

    def fit(self, X, y):
        bins_indexes = np.digitize(X, self.bins)
        for bin_index in range(1, self.m + 1):
            self.bins_means[bin_index] = y[bins_indexes == bin_index].mean()
        return self

    def predict(self, X):
        bins_indexes = np.digitize(X, self.bins)
        return self.bins_means[bins_indexes]


class LinearSmoother:
    def loo_score(self):
        X = self.X_.reshape(-1, 1)
        num = self.y_.reshape(-1, 1) - self.predict(X)
        den = 1 - np.diag(self.get_weights(X)).reshape(-1, 1)
        fraction = np.full_like(den, np.inf)
        fraction = np.divide(num, den, out=fraction, where=(den != 0))
        return (np.square(fraction)).mean()


class LocalPolynomials(LinearSmoother):
    def __init__(self, h, kernel, p):
        self.h = h
        self.kernel = kernel
        self.p = p

    def fit(self, X, y):
        self.X_ = X.reshape(1, -1)
        self.y_ = y
        return self

    def predict(self, X):
        distances = np.expand_dims(X - self.X_, axis=-1)
        X_x = distances ** np.arange(self.p + 1)
        kernel_values = self.kernel(distances / self.h)
        weighted_X_x_T = (X_x * kernel_values).transpose((0, 2, 1))
        beta = np.linalg.inv(weighted_X_x_T @ X_x) @ weighted_X_x_T @ self.y_
        return beta[:, 0, :]


class NadarayaWatson(LocalPolynomials):
    def __init__(self, h, kernel):
        self.h_ = h
        self.kernel = kernel

    def fit(self, X, y):
        self.X_ = X.reshape(1, -1)
        self.y_ = y.reshape(1, -1)
        return self

    def get_weights(self, X):
        distances = X - self.X_
        kernel_values = self.kernel(distances / self.h_)
        weights = kernel_values / kernel_values.sum(axis=1, keepdims=True)
        return weights

    def predict(self, X):
        weights = self.get_weights(X)
        return (weights * self.y_).sum(axis=1, keepdims=True)


class GaussNW(NadarayaWatson):
    def __init__(self, h):
        self.h_ = h
        self.kernel = norm.pdf


class LocalAverages(NadarayaWatson):
    def __init__(self, h):
        self.h_ = h
        self.kernel = lambda x: np.abs(x) <= 1
