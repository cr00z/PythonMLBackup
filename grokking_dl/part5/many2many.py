import numpy as np


weights = np.array([[0.1, 0.1, -0.3],
                    [0.1, 0.2, 0.0],
                    [0.0, 1.3, 0.1]])

def neural_network(input, weights):
    assert len(input) == len(weights)
    output = [np.dot(input, weights[i]) for i in range(len(weights))]
    return output


toes = [8.5]
wlrec = [0.65]
nfans = [1.2]
hurt = [0.1]
win = [1]
sad = [0.1]
alpha = 0.01
input = np.array([toes[0], wlrec[0], nfans[0]])
true = np.array([hurt[0], win[0], sad[0]])
error = np.zeros(3)
delta = np.zeros(3)

pred = neural_network(input, weights)

error = (pred - true) ** 2
delta = pred - true
weights_deltas = np.array([
    [input[i] * delta[j] for j in range(len(delta))]
    for i in range(len(input))
])

print("iter: {}, pred: {}, err: {}, delta: {}, weights: {}, w_deltas: {}"
      .format(0, pred, error, delta, weights, weights_deltas))

