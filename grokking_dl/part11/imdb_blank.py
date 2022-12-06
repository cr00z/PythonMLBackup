# SECTION 1: one-hot encoding
import random
from collections import Counter
import numpy as np

np.random.seed(1)
random.seed(1)
with open('reviews.txt') as reviews:
    text = [line.upper() for line in reviews.readlines()]

tokens = list(map(lambda x: x.split(' '), text))

wordcnt = Counter()
for sent in tokens:
    for word in sent:
        wordcnt[word] -= 1
vocab = list(set(map(lambda x: x[0], wordcnt.most_common())))

word2index = {word: i for i, word in enumerate(vocab)}  # 74075

concatenated = list()
input_dataset = list()
for sent in tokens:
    sent_indices = list()
    for word in sent:
        sent_indices.append(word2index[word])
        concatenated.append(word2index[word])
    input_dataset.append(sent_indices)
concatenated = np.array(concatenated)
random.shuffle(input_dataset)


# SECTION 2: neural network

ALPHA = 0.05
ITERATIONS = 2
HIDDEN_SIZE = 50
WINDOW = 2
NEGATIVE = 5

weights_0_1 = 0.2 * np.random.random((len(vocab), HIDDEN_SIZE)) - 0.1
weights_1_2 = np.random.random((len(vocab), HIDDEN_SIZE)) * 0

layer2_target = np.zeros(NEGATIVE + 1)
layer2_target[0] = 1


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


for rev_i, review in enumerate(input_dataset * ITERATIONS):
    for target_i in range(len(review)):
        indexes = (np.random.rand(NEGATIVE) * len(concatenated)).astype('int').tolist()
        target_samples = [review[target_i]] + list(concatenated[indexes])
        left_context = review[max(0, target_i - WINDOW): target_i]
        right_context = review[target_i + 1: min(len(review), target_i + WINDOW)]

        layer1 = np.mean(weights_0_1[left_context + right_context], axis=0)
        layer2 = sigmoid(np.dot(layer1, weights_1_2[target_samples].T))

        layer2_delta = layer2 - layer2_target
        layer1_delta = layer2_delta.dot(weights_1_2[target_samples])

        weights_0_1[left_context + right_context] -= ALPHA * layer1_delta
        weights_1_2[target_samples] -= ALPHA * np.outer(layer2_delta, layer1)


# SECTION 3: Word Embeddings

import math


def similar(target='beautiful'):
    scores = Counter()
    target_index = word2index[target.upper()]
    for word, index in word2index.items():
        raw_difference = weights_0_1[index] - weights_0_1[target_index]
        squared_difference = raw_difference * raw_difference
        scores[word] = -math.sqrt(sum(squared_difference))
    return scores.most_common(10)


#print(similar('beautiful'))
#print(similar('terrible'))


# SECTION 4: Analogy

def analogy(positive=['terrible', 'good'], negative=['bad']):
    norms = np.sum(weights_0_1 * weights_0_1, axis=1)
    norms.resize(norms.shape[0], 1)

    normed_weights = weights_0_1 * norms

    query_vect = np.zeros(len(weights_0_1[0]))
    for word in positive:
        query_vect += normed_weights[word2index[word.upper()]]
    for word in negative:
        query_vect -= normed_weights[word2index[word.upper()]]

    scores = Counter()
    for word, index in word2index.items():
        raw_difference = weights_0_1[index] - query_vect
        squared_difference = raw_difference * raw_difference
        scores[word] = -math.sqrt(sum(squared_difference))

    return scores.most_common(10)[1:]


print(analogy(['terrible', 'good'], ['bad']))
print(analogy(['elizabeth', 'he'], ['she']))
