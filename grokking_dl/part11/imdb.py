# SECTION 1: one-hot encoding
import random

WORD_FREQ = 2

with open('labels.txt') as labels:
    # 25000
    target_dataset = [1 if lbl[0] == 'p' else 0 for lbl in labels.readlines()]

with open('reviews.txt') as reviews:
    text = [line.upper() for line in reviews.readlines()]
words = set(' '.join(text).split(' '))  # 74075

# often words to index (no less than WORD_FREQ times)
# words_temp = dict.fromkeys(words, 0)
# for word in text:
#     words_temp[word] += 1  # 74073
# often_words = [word for word, count in words_temp.items() if count >= WORD_FREQ]
# word2index = {word: i for i, word in enumerate(often_words)}
word2index = {word: i for i, word in enumerate(words)}  # 74075

input_dataset = list()
for line in text:
    words_in_review = set(line.split(' '))
    indexes = list()
    for word in words_in_review:
        indexes.append(word2index[word])
    input_dataset.append(indexes)

# print(len(input_dataset))
# print(text[0])
# print(sorted(input_dataset[0]))
# out = list()
# for word in set(text[0].split(' ')):
#     out.append(word2index[word])
# print(sorted(out))

# SECTION 2: neural network

import numpy as np
np.random.seed(1)


ALPHA = 0.01
ITERATIONS = 5
HIDDEN_SIZE = 100


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


weights_0_1 = 0.2 * np.random.random((len(words), HIDDEN_SIZE)) - 0.1
weights_1_2 = 0.2 * np.random.random((HIDDEN_SIZE, 1)) - 0.1

for iter in range(ITERATIONS):
    correct = 0
    total = 0

    for i in range(len(input_dataset) - 1000):
        layer0 = input_dataset[i]
        layer1 = sigmoid(np.sum(weights_0_1[layer0], axis=0))
        layer2 = sigmoid(np.dot(layer1, weights_1_2))

        layer2_delta = layer2 - target_dataset[i]
        layer1_delta = layer2_delta.dot(weights_1_2.T)

        weights_0_1[layer0] -= ALPHA * layer1_delta
        weights_1_2 -= ALPHA * np.outer(layer1, layer2_delta)

        if np.abs(layer2_delta) < 0.5:
            correct += 1
        total += 1

    test_correct = 0
    test_total = 0
    for i in range(len(input_dataset) - 1000, len(input_dataset)):
        layer0 = input_dataset[i]
        layer1 = sigmoid(np.sum(weights_0_1[layer0], axis=0))
        layer2 = sigmoid(np.dot(layer1, weights_1_2))

        if np.abs(layer2 - target_dataset[i]) < 0.5:
            test_correct += 1
        test_total += 1

    print("Iter: {}, train: {}, test: {}"
          .format(iter, correct / total, test_correct / test_total))

# SECTION 3: Word Embeddings

from collections import Counter
import math


def similar(target='beautiful'):
    scores = Counter()
    target_index = word2index[target.upper()]
    for word, index in word2index.items():
        raw_difference = weights_0_1[index] - weights_0_1[target_index]
        squared_difference = raw_difference * raw_difference
        scores[word] = -math.sqrt(sum(squared_difference))
    return scores.most_common(10)


print(similar('beautiful'))
print(similar('terrible'))
