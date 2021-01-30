import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from matplotlib.colors import ListedColormap


def data_shuffle(data):
    data_shuffled = shuffle(data, random_state=42)
    X = data_shuffled.drop('target', 1)
    y = data_shuffled['target']
    return data_shuffled, X, y


def argmin_value(x, w):
    """
    This function calculates the value to update the weights
    Args:
        x: np.array with transformed values
        w: np.array of weights
    Define input and expected output:
    >>> argmin_value(np.array([0, 0, 0, 1, 0, 0, 0]), np.array([1, 0, 0, 0, 0, 0, 0]))
    0.4999999999999999
    """
    return max(-x.dot((w-x).T)/(np.linalg.norm(w-x))**2, 0)


def update_weights(q, x, w):
    return q*w + (1-q)*x


def vectors_transformation(x):
    """
    This function transform array of input values into another array, using specific rule
    Define input and expected output:
    >>> vectors_transformation(np.array([1, -1]))
    array([ 1, -1, -1,  1,  1, -1,  1])
    """
    return np.array([x[0]**2, x[0]*x[1], x[1]*x[0], x[1]**2, x[0], x[1], 1])


def initialization(dim):
    return np.zeros((dim, 7))


def check_eig(w):
    """
    This function calculates the matrix from weights, calculates its eig values and vectors
    and calculates the multiplication of weights and eig vectors
    >>> w_eta_l1, w_eta_l2, eta_l1, eta_l2 = check_eig(np.array([0.5, 1, 1, 0.5, 0, 0, 0]))
    >>> print(w_eta_l1, w_eta_l2)
    1.4999999999999996 -0.49999999999999983
    """
    A = np.array([[w[0], w[1]],
                 [w[2], w[3]]])
    v, l = np.linalg.eig(A)
    l1 = l[:, 0]
    l2 = l[:, 1]
    eta_l1 = np.array([l1[0]*l1[0], l1[0]*l1[1], l1[1]*l1[0], l1[1]*l1[1], 0, 0, 0])
    eta_l2 = np.array([l2[0]*l2[0], l2[0]*l2[1], l2[1]*l2[0], l2[1]*l2[1], 0, 0, 0])
    w_eta_l1 = np.dot(w, eta_l1)
    w_eta_l2 = np.dot(w, eta_l2)
    return w_eta_l1, w_eta_l2, eta_l1, eta_l2


class Kozinec(object):
    """
    This class represents Kozinec algorithm for finding the dividing surface in the form of ellipses.
    Kozinec algorithm is one of the algorithms for linear separation of finite sets of points.
    """
    def __init__(self, n_iter=10):
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w = np.array([0.5, 0., 0., 0.5, 0., 0., 0.])
        #self.w = vectors_transformation(X[5])
        ksi = initialization(X.shape[0])
        for i in range(self.n_iter):
            j = 0
            for xi, target in zip(X, y):
                ksi_i = vectors_transformation(xi)
                ksi[j] = ksi_i
                w_ksi = np.dot(self.w, ksi[j])
                if target == 1 and w_ksi >= 0:
                    ksi[j] = -ksi[j]
                    q = argmin_value(ksi[j], self.w)
                    self.w = update_weights(q, ksi[j], self.w)
                elif target == 0 and w_ksi <= 0:
                    q = argmin_value(ksi[j], self.w)
                    self.w = update_weights(q, ksi[j], self.w)
                w_eta_l1, w_eta_l2, eta_l1, eta_l2 = check_eig(self.w)
                if w_eta_l1 <= 0:
                    q = argmin_value(eta_l1, self.w)
                    self.w = update_weights(q, eta_l1, self.w)
                if w_eta_l2 <= 0:
                    q = argmin_value(eta_l2, self.w)
                    self.w = update_weights(q, eta_l2, self.w)
                j += 1           
        return self

    def input_values(self, X):
        return np.dot(X, self.w)

    def predict(self, X):
        ksi_pred = initialization(X.shape[0])
        k = 0
        for xi in X:
            ksi_pred_i = vectors_transformation(xi)
            ksi_pred[k] = ksi_pred_i
            k += 1
        return np.where(self.input_values(ksi_pred) >= 0.0, 0, 1)


def plot_decision_regions(X, y, classifier, resolution=0.01):
    # set up markers and colors
    markers = ('x', 'o')
    colors = ('purple', 'red')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # return a meshgrid of solution
    x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.transpose(np.array([xx1.ravel(), xx2.ravel()])))
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # show results on the plot
    for idx, c1 in enumerate(np.unique(y)):
        plt.scatter(x=X[y == c1, 0], y=X[y == c1, 1], alpha=0.8, color=cmap(idx), marker=markers[idx], label=c1)
