import numpy as np


weights = np.array([0.1, 0.2, -.1])


def neural_network(input, weights):
    assert len(input) == len(weights)
    return np.dot(input, weights)


toes = [8.5]
wlrec = [0.65]
nfans = [1.2]
win_or_lose_binary = [1]
true = win_or_lose_binary[0]
input = np.array([toes[0], wlrec[0], nfans[0]])
alpha = 0.01

for iter in range(3):
    pred = neural_network(input, weights)
    error = (pred - true) ** 2
    delta = pred - true
    weights_deltas = delta * input
    # weights_deltas[0] = 0
    print("iter: {}, pred: {:.4}, err: {:.6}, delta: {:.6}, weights: {}, w_deltas: {}"
          .format(iter+1, pred, error, delta, weights, weights_deltas))
    weights -= alpha * weights_deltas
