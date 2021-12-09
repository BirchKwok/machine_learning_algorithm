import numpy as np


class NaivePerceptron:
    """原始形式"""
    def __init__(self, random_state=None):
        self.weight = None
        self.bias = None
        self.random_state = np.random.seed(random_state)

    # define sign function
    @staticmethod
    def _lin(w, b, X):
        return np.dot(X, w) + b

    def _sign(self, w, b, X):
        return np.where(self._lin(w, b, X) >= 0, 1, -1)

    @staticmethod
    def gradient_descent(w, b, x_i, y_i, lr=0.01):
        # batch gradient descent
        w += lr * x_i * y_i
        b += lr * y_i
        return w, b

    def fit(self, X, y, learning_rate=0.01):
        assert isinstance(X, np.ndarray) and X.ndim == 2

        # initial weight and bias
        self.weight = np.zeros(X.shape[1])
        self.bias = 0

        wrong_sample = 1
        while wrong_sample > 0:
            for x_i, y_i in zip(X, y):
                # compute every sample distance, -y_i(wX+b)>=0
                dis = -y_i * self._lin(self.weight, self.bias, x_i)
                if dis >= 0:
                    self.weight, self.bias = self.gradient_descent(self.weight, self.bias,
                                                                   x_i, y_i, lr=learning_rate)
                    break
                wrong_sample = 0
            continue

        return self

    def predict(self, X):
        assert self.weight is not None and self.bias is not None, "model not fitted yet."
        return self._sign(self.weight, self.bias, X)


class Perceptron:
    """对偶形式，好处是提前计算好了相关矩阵，减少了计算量"""
    def __init__(self):
        self.weight = None
        self.bias = None

    @staticmethod
    def _compute_gram(X):
        return np.dot(X, np.transpose(X))

    # define sign function
    @staticmethod
    def _lin(w, b, X):
        return np.dot(X, w) + b

    def _sign(self, w, b, X):
        return np.where(self._lin(w, b, X) >= 0, 1, -1)

    def fit(self, X, y, learning_rate=1):
        # gram矩阵
        gram = self._compute_gram(X)

        # 初始化 alpha 和bias
        alpha = np.zeros(X.shape[0]) * learning_rate

        self.bias = 0

        wrong_sample = 1
        while wrong_sample > 0:
            for i in range(X.shape[0]):
                # y(\sum_{j=1}^{N}\alpha_j y_j x_j \cdot x_i + b)
                if (y[i] * (np.dot(alpha * y, gram[i]) + self.bias)) <= 0:
                    alpha[i] += learning_rate
                    self.bias += learning_rate * y[i]
                    break
                wrong_sample = 0

        self.weight = np.dot(alpha, X)

        return self

    def predict(self, X):
        assert self.weight is not None and self.bias is not None, "model not fitted yet."
        return self._sign(self.weight, self.bias, X)






















