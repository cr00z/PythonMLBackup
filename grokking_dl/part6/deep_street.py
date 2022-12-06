import numpy as np


np.random.seed(1)
NUM_OF_ITERATIONS = 60


def relu(x):
    return (x > 0) * x


def relu2deriv(output):
    return output > 0


alpha = 0.2
hidden_size = 4
streetlights = np.array([[1, 0, 1],
                         [0, 1, 1],
                         [0, 0, 1],
                         [1, 1, 1]])
walk_vs_stop = np.array([1, 1, 0, 0]).T
#weights_0_1 = 2 * np.random.random((3, hidden_size)) - 1
#weights_1_2 = 2 * np.random.random((hidden_size, 1)) - 1
#weights_0_1 = (np.random.random((3, hidden_size)) > 0.5) - 0.6
#weights_1_2 = (np.random.random((hidden_size, 1)) > 0.5) - 0.6
weights_0_1 = 2 * np.random.random((3, hidden_size)) - 1
weights_1_2 = 2 * np.random.random((hidden_size, 1)) - 1

for iteration in range(NUM_OF_ITERATIONS):
    layer_2_error = 0
    for i in range(len(streetlights)):
        layer_0 = streetlights[i:i+1]
        layer_1 = relu(np.dot(layer_0, weights_0_1))
        layer_2 = np.dot(layer_1, weights_1_2)

        layer_2_error += np.sum((layer_2 - walk_vs_stop[i:i+1]) ** 2)

        layer_2_delta = walk_vs_stop[i:i+1] - layer_2
        layer_1_delta = layer_2_delta.dot(weights_1_2.T) * relu2deriv(layer_1)

        weights_1_2_delta = layer_1.T.dot(layer_2_delta)
        weights_0_1_delta = layer_0.T.dot(layer_1_delta)

        print("layer_0: {}, layer_1: {}, layer_2: {}".format(layer_0, layer_1, layer_2))
        print("weight_1_2: {}, weight_0_1: {}".format(weights_1_2, weights_0_1))
        print("layer_2_delta: {}, layer_1_delta: {}".format(layer_2_delta, layer_1_delta))
        print("tw_1_2_delta: {}, w_0_1_delta: {}".format(weights_1_2_delta, weights_0_1_delta))
        print()

        weights_1_2 += alpha * weights_1_2_delta
        weights_0_1 += alpha * weights_0_1_delta
        input()
    #if iteration % 10 == 9:
    #    print("Error: {}".format(layer_2_error))
