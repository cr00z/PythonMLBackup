WEIGHTS = [0.3, 0.2, 0.9]


def neural_network(input_data, weight):
    prediction = [input_data * weight[i] for i in range(len(weight))]
    return prediction


toes = [8.5, 9.5, 9.9, 9.0]
wlrec = [0.65, 0.8, 0.8, 0.9]
nfans = [1.2, 1.3, 0.5, 1.0]


input_data = wlrec[0]
pred = neural_network(input_data, WEIGHTS)
print(pred)
