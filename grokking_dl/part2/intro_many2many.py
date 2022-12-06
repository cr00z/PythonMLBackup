WEIGHTS = [
    [0.1, 0.1, -0.3],
    [0.1, 0.2, 0.0],
    [0.0, 1.3, 0.1],
]


def neural_network(input_data, weights):
    prediction = []
    for weight in weights:
        assert(len(input_data) == len(weight))
        prediction.append(
            sum([input_data[i] * weight[i] for i in range(len(weight))])
        )
    return prediction


toes = [8.5, 9.5, 9.9, 9.0]
wlrec = [0.65, 0.8, 0.8, 0.9]
nfans = [1.2, 1.3, 0.5, 1.0]


input_data = [toes[0], wlrec[0], nfans[0]]
pred = neural_network(input_data, WEIGHTS)
print(pred)


import numpy as np
print([np.array(input_data).dot(weight) for weight in np.array(WEIGHTS)])
