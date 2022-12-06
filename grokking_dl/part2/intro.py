WEIGHT = 0.1


def neural_network(input_data, weight):
    prediction = input_data * weight
    return prediction


number_of_toes = [8.5, 9.5, 10, 9]
input_data = number_of_toes[0]
pred = neural_network(input_data, WEIGHT)
print(pred)
