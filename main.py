import numpy as np
import tensorflow as tf
import cnn
from utils import *
import loader
from math import ceil

def main():

    # Extract data from the csv

    data, n_labels = loader.data_extractor()
    train_data, train_labels, test_data, test_labels = get_train_test_data(data)

    # Build CNN model

    tf_data = tf.placeholder(tf.float32, [None, WIDTH * HEIGHT])
    tf_labels = tf.placeholder(tf.float32, [None, n_labels])
    features = tf.reshape(tf_data, [-1, 96, 96, 1])

    model = cnn.CNN_model(features, n_labels, tf_labels, [16, 32])

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    test_dict = {
        tf_data: test_data,
        tf_labels: test_labels
    }

    # Training

    rmse = [0] * EPOCHS

    for i in range(EPOCHS):
        train_data, train_labels = shuffle(train_data, train_labels)
        for j in range(train_data.shape[0] // BATCH_SIZE):
            batch_x = train_data[j * BATCH_SIZE:(j + 1) * BATCH_SIZE]
            batch_labels = train_labels[j * BATCH_SIZE:(j + 1) * BATCH_SIZE, :]

            feed_dict = {
                tf_data: batch_x,
                tf_labels: batch_labels
            }

            _ , train_loss = sess.run([model.optimizer, model.loss],
                                      feed_dict=feed_dict)

        test, rmse[i] = sess.run([model.predict, model.loss],
                                   feed_dict=test_dict)
        print('%3d Epoch, %.5f train loss, %.5f test_loss'
              % (i, train_loss, rmse[i]))

    # Plot the predictions for the first test image

    plot_img(test_data[0:1, :], test[0], n_labels)

    plt.plot(rmse)
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.savefig('learningcurve.png')
    plt.show()

if __name__ == '__main__':
    main()
