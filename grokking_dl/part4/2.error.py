weight = 0.5
input = 0.5
goal_pred = 0.8
step_amount = 0.001


def neural_network(input, weight):
    prediction = input * weight
    return prediction


for iteration in range(1101):
    pred = neural_network(input, weight)
    error = (pred - goal_pred) ** 2
    print("Error: {}, prediction: {}".format(error, pred))

    pred_up = neural_network(input, weight + step_amount)
    error_up = (pred_up - goal_pred) ** 2

    pred_dn = neural_network(input, weight - step_amount)
    error_dn = (pred_dn - goal_pred) ** 2

    if error_up < error_dn:
        weight += step_amount
    if error_dn < error_up:
        weight -= step_amount
