import numpy as np
weights = np.array([0.5, 0.48, -0.7])
alpha = 0.1
streetlights = np.array([[1, 0, 1],
                         [0, 1, 1],
                         [0, 0, 1],
                         [1, 1, 1],])
                         #[0, 1, 1],
                         #[1, 0, 1]])
#walk_vs_stop = np.array([0, 1, 0, 1, 1, 0]).T
walk_vs_stop = np.array([1, 1, 0, 0]).T

for iteration in range(100):
    error_for_all_lights = 0
    for row_index in range(len(walk_vs_stop)):
        inputs = streetlights[row_index]
        old_weight = weights
        goal_prediction = walk_vs_stop[row_index]
        prediction = inputs.dot(weights)
        error = (prediction - goal_prediction) ** 2
        error_for_all_lights += error
        delta = prediction - goal_prediction
        weights_delta = alpha * inputs * delta
        weights = weights - weights_delta
        print("input: {}, weight: {}, Pred: {}, delta: {}, weight_d: {}, weights: {}"
              .format(inputs, old_weight, prediction, delta, weights_delta, weights))
    #a = int(input())
