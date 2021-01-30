import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.utils import shuffle
from sklearn import metrics
import zipfile


def get_data(numbers):
    """
    This function returns train and test data and amount of different classes
    Args:
        numbers: list of numbers which we want to explore
    """
    numbers = numbers
    n_classes = len(numbers)
    z = zipfile.ZipFile('lab3/mnist.pkl.zip', 'r')
    k = z.extract('mnist.pkl')  # Извлечь файл из архива
    with open(k, 'rb') as f:
        train_set, _, test_set = pickle.load(f, encoding="bytes")
    x_train = train_set[0]
    x_test = test_set[0]
    x_train[x_train >= 0.5] = 1
    x_train[x_train < 0.5] = 0
    x_test[x_test >= 0.5] = 1
    x_test[x_test < 0.5] = 0
    y_train = train_set[1]
    y_test = test_set[1]
    idx_train = [[np.where(y_train == i)] for i in numbers]
    idx_test = [[np.where(y_test == i)] for i in numbers]
    idx_x_train = [x_train[idx_train[i][0]] for i in range(len(idx_train))]
    idx_x_test = [x_test[idx_test[i][0]] for i in range(len(idx_test))]
    idx_y_test = [y_test[idx_test[i][0]] for i in range(len(idx_test))]
    x_train_new = shuffle(np.concatenate(idx_x_train))
    x_test_new = shuffle(np.concatenate(idx_x_test))
    y_test_new = shuffle(np.concatenate(idx_y_test))
    return x_train_new, x_test_new, y_test_new, numbers, n_classes


def plot_image(data, n_classes):
    for i in range(n_classes):
        img = np.array(data)[i]
        img_reshape = img.reshape((28, 28))
        plt.imshow(img_reshape, cmap='gray')
        plt.show()


def init_parameters(data, n_classes):
    """
    This function calculates init p(k|x) and p(k)
    Args:
        data: np.array(N, D), D = 28*28
        n_classes: amount of different classes
    Returns:
        p_k_x_res: np.array(n_classes, D) - p0(k|x)
        init_p_k: np.array(n_classes) - init probabilities of each class
    >>> x_train, x_test, y_test, numbers, n_classes = get_data([0, 1, 2])
    >>> p_k_x_test, init_p_k_test = init_parameters(x_train, n_classes)
    >>> init_p_k_test.sum(axis=0)
    1.0
    """
    N = data.shape[0]
    K = n_classes
    D = data.shape[1]

    init_p_k_x = np.zeros((K,))

    matrix = np.random.rand(N, K)
    matrix /= matrix.sum(axis=1)[:, np.newaxis]
    for i in range(K):
        init_p_k_x[i] = sum(matrix[:, i]) / matrix.shape[0]
    p_k_x_res = np.repeat(init_p_k_x, 28 * 28).reshape(K, D)

    init_p_k = np.random.uniform(0, 1, K)
    init_p_k /= np.sum(init_p_k)

    return p_k_x_res, init_p_k


def m_step(data, p_k_x):
    """
    This function performs Maximization Step of the EM-algorithm
    Args:
        data: np.array(N, D), D = 28*28
        p_k_x: np.array(n_classes, D) - p(k|x)
    Returns:
        Nk/N: np.array(n_classes) - new p(k)
        p_i_j_new: np.array(n_classes, D)
    >>> test_pk, _ = m_step(np.array([[0,1], [1,0]]), np.array([[0.25, 0.75], [0.355, 0.645]]))
    >>> test_pk
    array([0.3025, 0.6975])
    >>> test_pk, _ = m_step(np.array([[0,1], [1,0]]), np.array([[0.11, 0.89], [0.225, 0.775]]))
    >>> test_pk
    array([0.1675, 0.8325])
    """
    N = data.shape[0]
    D = data.shape[1]
    K = p_k_x.shape[1]

    Nk = np.sum(p_k_x, axis=0)
    p_i_j_new = np.empty((K, D))

    for k in range(K):
        p_i_j_new[k] = np.sum(p_k_x[:, k][:, np.newaxis] * data, axis=0) / Nk[k]

    return Nk / N, p_i_j_new


def e_step(data, p_k, p_i_j):
    """
    This function performs Expectation Step of the EM-algorithm
    Args:
        data: np.array(N, D), D = 28*28
        p_k: np.array(n_classes)
        p_i_j: np.array(n_classes, D)
    Returns:
        p_k_x: np.array(n_classes, D) - new p(k|x)
    >>> test_e = e_step(np.array([[0,1], [1,0]]), np.array([0.1675, 0.8325]), np.array([[0.67164179, 0.32835821],[0.46546547, 0.53453453]]))
    >>> test_e
    array([[0.07056568, 0.92943432],
           [0.29523861, 0.70476139]])
    """
    N = data.shape[0]
    K = p_i_j.shape[0]

    p_k_x = np.empty((N, K))
    for i in range(N):
        for k in range(K):
            p_k_x[i, k] = np.prod((p_i_j[k] ** data[i]) * ((1 - p_i_j[k]) ** (1 - data[i])))
    p_k_x *= p_k

    p_k_x /= p_k_x.sum(axis=1)[:, np.newaxis]

    return p_k_x


def bernoulli_em_algorithm(data, n_classes, max_iters=100):
    """
    This function performs EM-algorithm
    Args:
        data: np.array(N, D), D = 28*28
        n_classes: amount of different classes
        max_iters: amount of iterations
    Returns:
        p_k: np.array(n_classes) - result probabilities of each class
        p_i_j: np.array(n_classes, D)
    """
    N = data.shape[0]
    D = data.shape[1]
    K = n_classes

    # initializing
    init_p_k_x, init_p_k = init_parameters(data, K)
    p_k_x = e_step(data, init_p_k, init_p_k_x)

    for i in range(max_iters):
        # perform M Step
        p_k, p_i_j = m_step(data, p_k_x)
        # perform E Step
        p_k_x = e_step(data, p_k, p_i_j)

    return p_k, p_i_j


def predict(data, labels, p_k, p_i_j, numbers):
    """
    This functions calculates predictions for test data and calculates mean squared error
    Args:
        data: np.array(N, D), D = 28*28
        labels: y_test
        p_k: np.array(n_classes) - result p(k) from the EM-algorithm on the train data
        p_i_j: np.array(n_classes, D) - result p_i_j from the EM-algorithm on the train data
        numbers: list of numbers that we want to see
    Returns:
        pred: np.array of the model predictions
    """
    pred = e_step(data, p_k, p_i_j).argmax(axis=1)
    for j in range(len(numbers)):
        for i in range(len(pred)):
            if pred[i] == j:
                pred[i] = numbers[j]
    print('metrics:', metrics.classification_report(labels, pred))
    return pred
