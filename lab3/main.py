from functions import get_data, plot_image, bernoulli_em_algorithm, predict
import numpy as np

x_train, x_test, y_test, numbers, n_classes = get_data([0, 1])
res_p_k, res_p_i_j = bernoulli_em_algorithm(x_train, n_classes, max_iters=10)

plot_image(res_p_i_j, n_classes)

with np.errstate(invalid='ignore'):
    predictions = predict(x_test, y_test, res_p_k, res_p_i_j, numbers)
