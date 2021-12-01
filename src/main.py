
import tensorflow as tf
import keras
from extra_keras_datasets import emnist

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np

import matplotlib.pyplot as plt

import os
import random

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

    # convert vectors to binary matrices
    y_train = tf.keras.utils.to_categorical(y_train, classes)
    y_test = tf.keras.utils.to_categorical(y_test, classes)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    return [x_train, y_train, x_test, y_test]


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
    # model creation (review this section. have it match the bottom string block)
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
    score = model.evaluate(x_test, y_test, verbose = 1)
    print('\nTest loss:', score[0])
    print('Test accuracy:', score[1])

    return


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


# testing and visualization of the metrics ==> visualize this as well, how many it got right and wrong
testing_visualization(t_t_data[2], t_t_data[3])

'''
End of Driver Code _____________________________________________________________________________________________________________________________
'''





# below is the original model analysis
'''

 Exec/report summary (analyze) 
 ------------------------------------------------------------------------------------------------------------------------ 
 model file           : /home/stm32mp1/Downloads/AI_Model_model-ABC123-112.h5 
 type                 : keras 
 c_name               : network 
 workspace dir        : /tmp/mxAI_workspace217353968906142404737480632270365 
 output dir           : /home/stm32mp1/.stm32cubemx 
  
 model_name           : AI_Model_modelABC123112 
 model_hash           : e1d8edfdd699bf1fe26e09d300d93711 
 input                : input_0 [784 items, 3.06 KiB, ai_float, float, (1, 28, 28, 1)] 
 inputs (total)       : 3.06 KiB 
 output               : dense_4_nl [36 items, 144 B, ai_float, float, (1, 1, 1, 36)] 
 outputs (total)      : 144 B 
 params #             : 74,508 items (291.05 KiB) 
 macc                 : 732,560 
 weights (ro)         : 298,032 B (291.05 KiB) 
 activations (rw)     : 23,584 B (23.03 KiB)  
 ram (total)          : 26,864 B (26.23 KiB) = 23,584 + 3,136 + 144 
  
 Model name - AI_Model_modelABC123112 ['input_0'] ['dense_4_nl'] 
 --------------------------------------------------------------------------------------------------------------------------------------------------------- 
 id   layer (type)                 oshape               param/size       macc      connected to      |   c_size   c_macc            c_type                
 --------------------------------------------------------------------------------------------------------------------------------------------------------- 
 0    input_0 (Input)              (h:28, w:28, c:1)                                                 |                              
      conv2d_2 (Conv2D)            (h:26, w:26, c:8)    80/320           48,680    input_0           |            +5,408(+11.1%)    conv2d()[0]           
      conv2d_2_nl (Nonlinearity)   (h:26, w:26, c:8)                     5,408     conv2d_2          |            -5,408(-100.0%)   
 --------------------------------------------------------------------------------------------------------------------------------------------------------- 
 1    conv2d_3 (Conv2D)            (h:24, w:24, c:8)    584/2,336        331,784   conv2d_2_nl       |            +9,216(+2.8%)     optimized_conv2d()[1] 
      conv2d_3_nl (Nonlinearity)   (h:24, w:24, c:8)                     4,608     conv2d_3          |            -4,608(-100.0%)   
 --------------------------------------------------------------------------------------------------------------------------------------------------------- 
 2    max_pooling2d_2 (Pool)       (h:12, w:12, c:8)                     4,608     conv2d_3_nl       |            -4,608(-100.0%)   
 --------------------------------------------------------------------------------------------------------------------------------------------------------- 
 3    conv2d_4 (Conv2D)            (h:10, w:10, c:16)   1,168/4,672      115,216   max_pooling2d_2   |            +1,600(+1.4%)     conv2d()[2]           
      conv2d_4_nl (Nonlinearity)   (h:10, w:10, c:16)                    1,600     conv2d_4          |            -1,600(-100.0%)   
 --------------------------------------------------------------------------------------------------------------------------------------------------------- 
 4    conv2d_5 (Conv2D)            (h:8, w:8, c:16)     2,320/9,280      147,472   conv2d_4_nl       |            +2,048(+1.4%)     optimized_conv2d()[3] 
      conv2d_5_nl (Nonlinearity)   (h:8, w:8, c:16)                      1,024     conv2d_5          |            -1,024(-100.0%)   
 --------------------------------------------------------------------------------------------------------------------------------------------------------- 
 5    max_pooling2d_3 (Pool)       (h:4, w:4, c:16)                      1,024     conv2d_5_nl       |            -1,024(-100.0%)   
 --------------------------------------------------------------------------------------------------------------------------------------------------------- 
 7    flatten_1 (Reshape)          (c:256)                                         max_pooling2d_3   |                              
 --------------------------------------------------------------------------------------------------------------------------------------------------------- 
 8    dense_3 (Dense)              (c:240)              61,680/246,720   61,680    flatten_1         |                              dense()[4]            
      dense_3_nl (Nonlinearity)    (c:240)                               240       dense_3           |                              nl()[5]               
 --------------------------------------------------------------------------------------------------------------------------------------------------------- 
 10   dense_4 (Dense)              (c:36)               8,676/34,704     8,676     dense_3_nl        |                              dense()[6]            
      dense_4_nl (Nonlinearity)    (c:36)                                540       dense_4           |                              nl()/o[7]             
 --------------------------------------------------------------------------------------------------------------------------------------------------------- 
 model/c-model: macc=732,560/732,560  weights=298,032/298,032  activations=--/23,584 io=--/3,280 
  
 Complexity report per layer - macc=732,560 weights=298,032 act=23,584 ram_io=3,280 
 --------------------------------------------------------------------------- 
 id   name         c_macc                    c_rom                     c_id 
 --------------------------------------------------------------------------- 
 0    conv2d_2     |||                7.4%   |                  0.1%   [0]  
 1    conv2d_3     ||||||||||||||||  46.5%   |                  0.8%   [1]  
 3    conv2d_4     ||||||            15.9%   |                  1.6%   [2]  
 4    conv2d_5     |||||||           20.4%   |                  3.1%   [3]  
 8    dense_3      |||                8.4%   ||||||||||||||||  82.8%   [4]  
 8    dense_3_nl   |                  0.0%   |                  0.0%   [5]  
 10   dense_4      |                  1.2%   |||               11.6%   [6]  
 10   dense_4_nl   |                  0.1%   |                  0.0%   [7]

'''




# below is what I have thus far
'''

  
 Exec/report summary (analyze) 
 ------------------------------------------------------------------------------------------------------------------------ 
 model file           : /home/stm32mp1/Robert/HCR/results/AI_Model_model-ABC123-112.h5 
 type                 : keras 
 c_name               : network_3 
 workspace dir        : /tmp/mxAI_workspace137168132698079260628899292301142 
 output dir           : /home/stm32mp1/.stm32cubemx 
  
 model_name           : AI_Model_modelABC123112 
 model_hash           : 0100ba309f36706e13e9945d825d29c8 
 input                : input_0 [784 items, 3.06 KiB, ai_float, float, (1, 28, 28, 1)] 
 inputs (total)       : 3.06 KiB 
 output               : dense_1_nl [36 items, 144 B, ai_float, float, (1, 1, 1, 36)] 
 outputs (total)      : 144 B 
 params #             : 74,508 items (291.05 KiB) 
 macc                 : 732,560 
 weights (ro)         : 298,032 B (291.05 KiB) 
 activations (rw)     : 23,584 B (23.03 KiB)  
 ram (total)          : 26,864 B (26.23 KiB) = 23,584 + 3,136 + 144 
  
 Model name - AI_Model_modelABC123112 ['input_0'] ['dense_1_nl'] 
 --------------------------------------------------------------------------------------------------------------------------------------------------------- 
 id   layer (type)                 oshape               param/size       macc      connected to      |   c_size   c_macc            c_type                
 --------------------------------------------------------------------------------------------------------------------------------------------------------- 
 0    input_0 (Input)              (h:28, w:28, c:1)                                                 |                              
      conv2d (Conv2D)              (h:26, w:26, c:8)    80/320           48,680    input_0           |            +5,408(+11.1%)    conv2d()[0]           
      conv2d_nl (Nonlinearity)     (h:26, w:26, c:8)                     5,408     conv2d            |            -5,408(-100.0%)   
 --------------------------------------------------------------------------------------------------------------------------------------------------------- 
 1    conv2d_1 (Conv2D)            (h:24, w:24, c:8)    584/2,336        331,784   conv2d_nl         |            +9,216(+2.8%)     optimized_conv2d()[1] 
      conv2d_1_nl (Nonlinearity)   (h:24, w:24, c:8)                     4,608     conv2d_1          |            -4,608(-100.0%)   
 --------------------------------------------------------------------------------------------------------------------------------------------------------- 
 2    max_pooling2d (Pool)         (h:12, w:12, c:8)                     4,608     conv2d_1_nl       |            -4,608(-100.0%)   
 --------------------------------------------------------------------------------------------------------------------------------------------------------- 
 3    conv2d_2 (Conv2D)            (h:10, w:10, c:16)   1,168/4,672      115,216   max_pooling2d     |            +1,600(+1.4%)     conv2d()[2]           
      conv2d_2_nl (Nonlinearity)   (h:10, w:10, c:16)                    1,600     conv2d_2          |            -1,600(-100.0%)   
 --------------------------------------------------------------------------------------------------------------------------------------------------------- 
 4    conv2d_3 (Conv2D)            (h:8, w:8, c:16)     2,320/9,280      147,472   conv2d_2_nl       |            +2,048(+1.4%)     optimized_conv2d()[3] 
      conv2d_3_nl (Nonlinearity)   (h:8, w:8, c:16)                      1,024     conv2d_3          |            -1,024(-100.0%)   
 --------------------------------------------------------------------------------------------------------------------------------------------------------- 
 5    max_pooling2d_1 (Pool)       (h:4, w:4, c:16)                      1,024     conv2d_3_nl       |            -1,024(-100.0%)   
 --------------------------------------------------------------------------------------------------------------------------------------------------------- 
 7    flatten (Reshape)            (c:256)                                         max_pooling2d_1   |                              
 --------------------------------------------------------------------------------------------------------------------------------------------------------- 
 8    dense (Dense)                (c:240)              61,680/246,720   61,680    flatten           |                              dense()[4]            
      dense_nl (Nonlinearity)      (c:240)                               240       dense             |                              nl()[5]               
 --------------------------------------------------------------------------------------------------------------------------------------------------------- 
 10   dense_1 (Dense)              (c:36)               8,676/34,704     8,676     dense_nl          |                              dense()[6]            
      dense_1_nl (Nonlinearity)    (c:36)                                540       dense_1           |                              nl()/o[7]             
 --------------------------------------------------------------------------------------------------------------------------------------------------------- 
 model/c-model: macc=732,560/732,560  weights=298,032/298,032  activations=--/23,584 io=--/3,280 
  
 Complexity report per layer - macc=732,560 weights=298,032 act=23,584 ram_io=3,280 
 --------------------------------------------------------------------------- 
 id   name         c_macc                    c_rom                     c_id 
 --------------------------------------------------------------------------- 
 0    conv2d       |||                7.4%   |                  0.1%   [0]  
 1    conv2d_1     ||||||||||||||||  46.5%   |                  0.8%   [1]  
 3    conv2d_2     ||||||            15.9%   |                  1.6%   [2]  
 4    conv2d_3     |||||||           20.4%   |                  3.1%   [3]  
 8    dense        |||                8.4%   ||||||||||||||||  82.8%   [4]  
 8    dense_nl     |                  0.0%   |                  0.0%   [5]  
 10   dense_1      |                  1.2%   |||               11.6%   [6]  
 10   dense_1_nl   |                  0.1%   |                  0.0%   [7] 
                                                     
'''