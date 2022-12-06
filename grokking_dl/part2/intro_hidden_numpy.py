import numpy as np


id_wgt = np.array([
    [0.1, 0.2, -0.1],
    [-0.1, 0.1, 0.9],
    [0.1, 0.4, 0.1],
]).T
hp_wgt = np.array([
    [0.3, 1.1, -0.3],
    [0.1, 0.2, 0.0],
    [0.0, 1.3, 0.1],
]).T
WEIGHTS = [id_wgt, hp_wgt]


def neural_network(input_data, weights):
    pr_hid = np.dot(input_data, weights[0])
    prediction = np.dot(pr_hid, weights[1])
    return prediction


toes = np.array([8.5, 9.5, 9.9, 9.0])
wlrec = np.array([0.65, 0.8, 0.8, 0.9])
nfans = np.array([1.2, 1.3, 0.5, 1.0])


input_data = np.array([toes[0], wlrec[0], nfans[0]])
pred = neural_network(input_data, WEIGHTS)
print(pred)
