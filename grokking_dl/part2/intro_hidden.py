id_wgt = [
    [0.1, 0.2, -0.1],
    [-0.1, 0.1, 0.9],
    [0.1, 0.4, 0.1],
]
hp_wgt = [
    [0.3, 1.1, -0.3],
    [0.1, 0.2, 0.0],
    [0.0, 1.3, 0.1],
]
WEIGHTS = [id_wgt, hp_wgt]


def vect_mat_mul(vect, matrix):
    result = []
    for row in matrix:
        assert(len(vect) == len(row))
        result.append(
            sum([vect[i] * row[i] for i in range(len(row))])
        )
    return result


def neural_network(input_data, weights):
    pr_hid = vect_mat_mul(input_data, weights[0])
    prediction = vect_mat_mul(pr_hid, weights[1])
    return prediction


toes = [8.5, 9.5, 9.9, 9.0]
wlrec = [0.65, 0.8, 0.8, 0.9]
nfans = [1.2, 1.3, 0.5, 1.0]


input_data = [toes[0], wlrec[0], nfans[0]]
pred = neural_network(input_data, WEIGHTS)
print(pred)
