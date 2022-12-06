import sys
import numpy as np
from keras.datasets import mnist
np.random.seed(1)


DATASET_SIZE = 1000
IMG_ROWS = 28
IMG_COLS = 28
IMG_SIZE = IMG_ROWS * IMG_COLS
IMG_COLORS = 255
NUM_LABELS = 10


(x_train, y_train), (x_test, y_test) = mnist.load_data()

images = x_train[0:DATASET_SIZE].reshape(DATASET_SIZE, IMG_SIZE) / IMG_COLORS
labels = np.zeros((DATASET_SIZE, NUM_LABELS))
for i, l in enumerate(y_train[0:DATASET_SIZE]):
    labels[i][l] = 1

test_images = x_test.reshape(len(x_test), IMG_SIZE) / IMG_COLORS
test_labels = np.zeros((len(y_test), NUM_LABELS))
for i, l in enumerate(y_test):
    test_labels[i][l] = 1


def tanh2deriv(output):
    return 1 - (output ** 2)


def softmax(x):
    temp = np.exp(x)
    return temp / np.sum(temp, axis=1, keepdims=True)


def get_image_section(layer, row_from, row_to, col_from, col_to):
    section = layer[:, row_from:row_to, col_from:col_to]
    return section.reshape(-1, 1, row_to-row_from, col_to-col_from)


ALPHA = 2
ITERATIONS = 300
BATCH_SIZE = 100
KERNEL_ROWS = 3
KERNEL_COLS = 3
NUM_KERNELS = 16
HIDDEN_SIZE = (IMG_ROWS - KERNEL_ROWS) * (IMG_COLS - KERNEL_COLS) * NUM_KERNELS

kernels = 0.02 * np.random.random(
    (KERNEL_ROWS * KERNEL_COLS, NUM_KERNELS)) - 0.01
weights_1_2 = 0.2 * np.random.random((HIDDEN_SIZE, NUM_LABELS)) - 0.1


for j in range(ITERATIONS):
    correct_cnt = 0

    for i in range(int(DATASET_SIZE / BATCH_SIZE)):
        batch_start = i * BATCH_SIZE
        batch_end = batch_start + BATCH_SIZE

        layer_0 = images[batch_start:batch_end]
        # -> (100, 28, 28)
        layer_0 = layer_0.reshape(layer_0.shape[0], IMG_ROWS, IMG_COLS)

        sections = list()
        for row_start in range(layer_0.shape[1] - KERNEL_ROWS):
            for col_start in range(layer_0.shape[2] - KERNEL_COLS):
                # -> (100, 1, 3, 3)
                section = get_image_section(layer_0,
                                            row_start,
                                            row_start + KERNEL_ROWS,
                                            col_start,
                                            col_start + KERNEL_COLS)
                # -> (625, 100, 1, 3, 3)
                sections.append(section)
        # -> (100, 625, 3, 3)
        expanded_input = np.concatenate(sections, axis=1)
        es = expanded_input.shape
        # -> (62500, 9)
        flattened_input = expanded_input.reshape(es[0] * es[1], -1)
        # -> (62500, 16)
        kernel_output = flattened_input.dot(kernels)
        # -> (100, 10000)
        layer_1 = np.tanh(kernel_output.reshape(es[0], -1))
        dropout_mask = np.random.randint(2, size=layer_1.shape)
        layer_1 *= dropout_mask * 2
        layer_2 = softmax(np.dot(layer_1, weights_1_2))

        for k in range(BATCH_SIZE):
            correct_cnt += int(
                np.argmax(layer_2[k:k + 1]) == np.argmax(
                    labels[batch_start + k:batch_start + k + 1])
            )

        layer_2_delta = (labels[batch_start:batch_end] - layer_2) / \
                        (BATCH_SIZE * layer_2.shape[0])
        layer_1_delta = layer_2_delta.dot(weights_1_2.T) * tanh2deriv(layer_1)
        layer_1_delta *= dropout_mask
        weights_1_2 += ALPHA * layer_1.T.dot(layer_2_delta)
        kernels -= ALPHA * flattened_input.T.dot(
            layer_1_delta.reshape(kernel_output.shape)
        )

    test_correct_cnt = 0

    for i in range(len(test_images)):
        layer_0 = test_images[i:i + 1]
        layer_0 = layer_0.reshape(layer_0.shape[0], IMG_ROWS, IMG_COLS)
        sections = list()
        for row_start in range(layer_0.shape[1] - KERNEL_ROWS):
            for col_start in range(layer_0.shape[2] - KERNEL_COLS):
                section = get_image_section(layer_0,
                                            row_start,
                                            row_start + KERNEL_ROWS,
                                            col_start,
                                            col_start + KERNEL_COLS)
                sections.append(section)
        expanded_input = np.concatenate(sections, axis=1)
        es = expanded_input.shape
        flattened_input = expanded_input.reshape(es[0] * es[1], -1)
        kernel_output = flattened_input.dot(kernels)
        layer_1 = np.tanh(kernel_output.reshape(es[0], -1))
        layer_2 = softmax(np.dot(layer_1, weights_1_2))

        test_correct_cnt += int(
            np.argmax(layer_2) == np.argmax(test_labels[i:i + 1])
        )

    print("Itr: {}, Train-Acc: {}, Test-Acc: {}"
          .format(j, correct_cnt / DATASET_SIZE,
                  test_correct_cnt / len(test_images))
          )
