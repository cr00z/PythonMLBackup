import sys
import numpy as np
from keras.datasets import mnist


np.random.seed(1)


BATCH_SIZE = 100
ALPHA = 2
ITERATIONS = 300
HIDDEN_SIZE = 100
NUM_LABELS = 10
DATASET_SIZE = 1000
IMG_X_SIZE = 28
IMG_Y_SIZE = 28
PIXELS_PER_IMAGE = IMG_X_SIZE * IMG_Y_SIZE
IMG_COLORS = 255


def tanh2deriv(output):
    return 1 - output ** 2


def softmax(x):
    temp = np.exp(x)
    return temp / np.sum(temp, axis=1, keepdims=True)


weights_0_1 = 0.02 * np.random.random((PIXELS_PER_IMAGE, HIDDEN_SIZE)) - 0.01
weights_1_2 = 0.2 * np.random.random((HIDDEN_SIZE, NUM_LABELS)) - 0.1

(x_train, y_train), (x_test, y_test) = mnist.load_data()

images = x_train[0:DATASET_SIZE]\
             .reshape(DATASET_SIZE, PIXELS_PER_IMAGE) / IMG_COLORS
labels = np.zeros((DATASET_SIZE, NUM_LABELS))
for i, l in enumerate(y_train[0:DATASET_SIZE]):
    labels[i][l] = 1

test_images = x_test.reshape(len(x_test), PIXELS_PER_IMAGE) / IMG_COLORS
test_labels = np.zeros((len(y_test), NUM_LABELS))
for i, l in enumerate(y_test):
    test_labels[i][l] = 1

for j in range(ITERATIONS):
    correct_cnt = 0

    for i in range(int(DATASET_SIZE / BATCH_SIZE)):
        batch_start = i * BATCH_SIZE
        batch_end = batch_start + BATCH_SIZE

        layer_0 = images[batch_start:batch_end]
        layer_1 = np.tanh(np.dot(layer_0, weights_0_1))
        dropout_mask = np.random.randint(2, size=layer_1.shape)
        layer_1 *= dropout_mask * 2
        layer_2 = softmax(np.dot(layer_1, weights_1_2))

        for k in range(BATCH_SIZE):
            correct_cnt += int(
                np.argmax(layer_2[k:k+1]) == np.argmax(labels[batch_start+k:batch_start+k+1])
            )

        layer_2_delta = (labels[batch_start:batch_end] - layer_2) /\
                        (BATCH_SIZE * layer_2.shape[0])
        layer_1_delta = layer_2_delta.dot(weights_1_2.T) * tanh2deriv(layer_1)
        layer_1_delta *= dropout_mask

        weights_1_2 += ALPHA * layer_1.T.dot(layer_2_delta)
        weights_0_1 += ALPHA * layer_0.T.dot(layer_1_delta)

    if (j % 10 == 0) or (j == ITERATIONS - 1):
        test_correct_cnt = 0

        for i in range(len(test_images)):
            layer_0 = test_images[i:i+1]
            layer_1 = np.tanh(np.dot(layer_0, weights_0_1))
            layer_2 = np.dot(layer_1, weights_1_2)

            test_correct_cnt += int(
                np.argmax(layer_2) == np.argmax(test_labels[i:i+1])
            )

        print("Itr: {}, Train-Acc: {}, Test-Acc: {}"
              .format(j, correct_cnt / DATASET_SIZE,
                      test_correct_cnt / len(test_images))
              )
