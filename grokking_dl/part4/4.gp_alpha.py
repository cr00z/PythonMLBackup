weight = 0.0
input = 2
goal_pred = 0.8


def neural_network(input, weight):
    prediction = input * weight
    return prediction


for iteration in range(20):
    pred = neural_network(input, weight)
    error = (pred - goal_pred) ** 2
    delta = pred - goal_pred
    weight_delta = delta * input
    weight = weight - weight_delta
    print("Error: {}, prediction: {}".format(error, pred))
    print("Delta: {}, weight delta: {}".format(delta, weight_delta))