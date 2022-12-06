weight = 0.5
input = 0.5
goal_pred = 0.8


def neural_network(input, weight):
    prediction = input * weight
    return prediction


for iteration in range(20):
    pred = neural_network(input, weight)
    error = (pred - goal_pred) ** 2
    direction_and_amount = (pred - goal_pred) * input
    weight = weight - direction_and_amount
    print("Error: {}, prediction: {}".format(error, pred))
