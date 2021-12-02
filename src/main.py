
import tensorflow as tf
import keras
from extra_keras_datasets import emnist

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import os
import random
import datetime

os.chdir('/home/stm32mp1/Robert/HCR/src') # can now run in any directory on the terminal. This changes the pwd at runtime
basepath = os.getcwd()[:-4]

if not os.path.isdir(f"{basepath}/results"):
    os.mkdir(f"{basepath}/results")

# model configurations
num_classes = 36 # 10 digits (0 - 9) + 26 letters (a - z)
batch_size = 256
epochs = 1
loss_function = "adam"
dropout_amount = 0.4
input_shape = (28, 28, 1)


def prepare_data(classes = num_classes):
    (dig_x_train, dig_y_train), (dig_x_test, dig_y_test) = emnist.load_data(type = 'mnist')
    (let_x_train, let_y_train), (let_x_test, let_y_test) = emnist.load_data(type = 'letters')
    # increment letter labels by + 9 in order to offset the digit labels already taking up the labels 0 - 9
    for i in range(0, len(let_y_train)):
        let_y_train[i] += 9
    for i in range(0, len(let_y_test)):
        let_y_test[i] += 9

    x_train = np.concatenate((let_x_train, dig_x_train), axis = 0)
    y_train = np.concatenate((let_y_train, dig_y_train), axis = 0)
    x_test = np.concatenate((let_x_test, dig_x_test), axis = 0)
    y_test = np.concatenate((let_y_test, dig_y_test), axis = 0)

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    
    unchanged_labels = y_test # keep original test labels before they get conv. to binary matrix

    # convert vectors to binary matrices
    y_train = tf.keras.utils.to_categorical(y_train, classes)
    y_test = tf.keras.utils.to_categorical(y_test, classes)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    return [x_train, y_train, x_test, y_test, unchanged_labels]


def visualize_data(x_train, y_train): 
    num = 25 # choose 25 samples
    starting_point = random.randint(0, len(x_train) - num) # start somewhere up until the end - num
    ending_point = starting_point + num + 1 # end #num samples after that point
    images = x_train[starting_point:ending_point].squeeze()
    labels = y_train[starting_point:ending_point]

    num_row = 5
    num_col = 5 
    fig, axes = plt.subplots(num_row, num_col, figsize=(1.5 * num_col, 2 * num_row))
    for i in range(num):
        ax = axes[i // num_col, i % num_col]
        ax.imshow(images[i], cmap='gray')
        ax.set_title(f'Label: {int(labels[i].nonzero()[0])}')
    plt.tight_layout()
    plt.show()
    

def create_model(shape = input_shape, dropout = dropout_amount, optim = loss_function, classes = num_classes):
    model = Sequential()
    model.add(Conv2D(8, kernel_size = (3, 3), activation = 'relu', input_shape = shape)) 
    model.add(Conv2D(8, (3, 3), activation = 'relu')) 
    model.add(MaxPooling2D(pool_size = (2, 2))) 
    model.add(Conv2D(16, (3, 3), activation = 'relu')) 
    model.add(Conv2D(16, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(dropout))
    model.add(Flatten())
    model.add(Dense(240, activation = 'relu')) 
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation = 'softmax')) 
    model.compile(loss = keras.losses.categorical_crossentropy, optimizer = optim, metrics = ['accuracy'])

    return model


def testing_visualization(x_test, y_test):
    y_pred = np.argmax(model.predict(x_test), axis = 1) # Predicted labels

    confusion_mtx = tf.math.confusion_matrix(y_test, y_pred, num_classes = num_classes) 
    plt.figure(figsize=(10, 8))
    ax = plt.axes()
    ax.set_title("Handwriting Character Recognition Prediction Results")
    sns.heatmap(confusion_mtx, annot = False, ax = ax)
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    plt.show()


'''
Driver Code ____________________________________________________________________________________________________________________________________
'''
model = create_model()
t_t_data = prepare_data()
visualize_data(t_t_data[0], t_t_data[1])

#training and testing, then saving
history = model.fit(t_t_data[0], t_t_data[1], batch_size = batch_size, epochs = epochs, verbose = 1, validation_data = (t_t_data[2], t_t_data[3]))
model_path = model.save(f'{basepath}/results/AI_Model_model-ABC123-112.h5')
print(f"Saving the model at {basepath}/results/AI_Model_model-ABC123-112.h5")

# Final Score in terminal
score = model.evaluate(t_t_data[2], t_t_data[3], verbose = 1)
print('\nTest loss:', score[0])
print('Test accuracy:', score[1])

# testing and visualization of the metrics
print("\nLaunching prediction accuracy heatmap...")
testing_visualization(t_t_data[2], t_t_data[4])

'''
End of Driver Code _____________________________________________________________________________________________________________________________
'''
