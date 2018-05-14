import numpy as np
from tkinter import *
import matplotlib.pyplot as plt
import math

HEIGHT = 96
WIDTH = 96
STRIDE = 2
FSIZE = [5, 5]
PSIZE = [2, 2]
BATCH_SIZE = 100
EPOCHS = 20
TEST_SIZE = 0.2

def shuffle(data, labels):
    order = np.random.permutation(data.shape[0])
    return data[order], labels[order]

def plot_img(data, labels, n_output):
    imgdata = data[0].reshape((WIDTH, HEIGHT)) / 255
    labels = labels * WIDTH
    x = labels[::2]
    y = labels[1::2]

    plt.imshow(imgdata, cmap='gray')
    plt.plot(x, y, 'ro')
    plt.savefig('prediction.png')
    plt.show()

def get_train_test_data(data):
    n_samples = data.shape[0]
    test_idx = math.ceil(n_samples * TEST_SIZE)

    train_data = np.array(data.iloc[test_idx:, -1].values)
    train_labels = np.array(data.iloc[test_idx:, :-1].values)

    test_data = np.array(data.iloc[:test_idx, -1].values)
    test_labels = np.array(data.iloc[:test_idx, :-1].values)

    train_data = np.array(train_data.tolist(), dtype=np.float32)
    test_data = np.array(test_data.tolist(), dtype=np.float32)

    return train_data, train_labels, test_data, test_labels
