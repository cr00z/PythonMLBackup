WEIGHTS = [0.1, 0.2, 0]


def elementwise_multiplication(vec_a, vec_b):
    assert (len(vec_a) == len(vec_b))
    return [vec_a[i] * vec_b[i] for i in range(len(vec_a))]


def elementwise_addition(vec_a, vec_b):
    assert (len(vec_a) == len(vec_b))
    return [vec_a[i] + vec_b[i] for i in range(len(vec_a))]
    pass


def vector_sum(vec_a):
    return sum(vec_a)


def vector_average(vec_a):
    return sum(vec_a) / len(vec_a)


def w_sum(add1, add2):
    # result = 0
    # for i in range(len(add1)):
    #     result += add1[i] * add2[i]
    # return result
    assert (len(add1) == len(add2))
    return sum(map(lambda x, y: x * y, add1, add2))


def neural_network(input_data, weight):
    prediction = w_sum(input_data, weight)
    return prediction


toes = [8.5, 9.5, 9.9, 9.0]
wlrec = [0.65, 0.8, 0.8, 0.9]
nfans = [1.2, 1.3, 0.5, 1.0]

input_data = [toes[0], wlrec[0], nfans[0]]
pred = neural_network(input_data, WEIGHTS)
print(pred)
print(vector_sum(elementwise_multiplication(input_data, WEIGHTS)))
print(vector_average([]))
