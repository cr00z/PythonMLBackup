weight = 0.1
lr = 0.01


def neural_network(input, weight):
    prediction = input * weight
    return prediction


number_of_toes = [8.5]
win_or_lose_binary = [1]


input = number_of_toes[0]
true = win_or_lose_binary[0]


pred = neural_network(input, weight)
error = (pred - true) ** 2
print(error)

pred_up = neural_network(input, weight + lr)
error_up = (pred_up - true) ** 2
print(error_up)

pred_dn = neural_network(input, weight - lr)
error_dn = (pred_dn - true) ** 2
print(error_dn)


if (error > error_up) or (error > error_dn):
    if error_up < error_dn:
        weight += lr
    if error_dn < error_up:
        weight -= lr

print(weight)