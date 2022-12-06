import numpy as np


weights = np.array([0.3, 0.2, 0.9])
wlrec = [0.65]
hurt = [0.1]
win = [1]
sad = [0.1]
input = wlrec[0]
true = np.array([hurt[0], win[0], sad[0]])
error = [0, 0, 0]
delta = [0, 0, 0]
alpha = 0.1


def neural_network(input, weights):
    return input * weights


pred = neural_network(input, weights)
error = (pred - true) ** 2
delta = pred - true
weights_deltas = delta * input

print("iter: {}, pred: {}, err: {}, delta: {}, weights: {}, w_deltas: {}"
      .format(0, pred, error, delta, weights, weights_deltas))

weights -= weights_deltas * alpha

print("iter: {}, pred: {}, err: {}, delta: {}, weights: {}, w_deltas: {}"
      .format(2, pred, error, delta, weights, weights_deltas))
