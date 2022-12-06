import sys
import numpy as np
from keras.datasets import mnist


np.random.seed(1)


ALPHA = 0.005
ITERATIONS = 350
HIDDEN_SIZE = 100#40
NUM_LABELS = 10
DATASET_SIZE = 1000
IMG_X_SIZE = 28
IMG_Y_SIZE = 28
PIXELS_PER_IMAGE = IMG_X_SIZE * IMG_Y_SIZE
IMG_COLORS = 255


relu = lambda x: (x > 0) * x
relu2deriv = lambda x: x > 0


weights_0_1 = 0.2 * np.random.random((PIXELS_PER_IMAGE, HIDDEN_SIZE)) - 0.1
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
    error = 0
    correct_cnt = 0

    for i in range(DATASET_SIZE):
        layer_0 = images[i:i+1]
        layer_1 = relu(np.dot(layer_0, weights_0_1))
        layer_2 = np.dot(layer_1, weights_1_2)

        error += np.sum((labels[i:i+1] - layer_2) ** 2)
        correct_cnt += int(np.argmax(layer_2) == np.argmax(labels[i:i+1]))

        layer_2_delta = labels[i:i+1] - layer_2
        layer_1_delta = layer_2_delta.dot(weights_1_2.T) * relu2deriv(layer_1)

        weights_1_2 += ALPHA * layer_1.T.dot(layer_2_delta)
        weights_0_1 += ALPHA * layer_0.T.dot(layer_1_delta)

    if (j % 10 == 0) or (j == ITERATIONS - 1):
        test_error = 0
        test_correct_cnt = 0

        for i in range(len(test_images)):
            layer_0 = test_images[i:i+1]
            layer_1 = relu(np.dot(layer_0, weights_0_1))
            layer_2 = np.dot(layer_1, weights_1_2)

            test_error += np.sum((test_labels[i:i+1] - layer_2) ** 2)
            test_correct_cnt += int(
                np.argmax(layer_2) == np.argmax(test_labels[i:i+1])
            )

        print("Itr: {}, Train-Err: {}, Train-Acc: {}, Test-Err: {}, Test-Acc: {}"
              .format(j, error / DATASET_SIZE, correct_cnt / DATASET_SIZE,
                      test_error / len(test_images),
                      test_correct_cnt / len(test_images)))
