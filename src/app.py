from keras import Sequential
from keras.datasets import cifar100
from keras.layers import Dense, Dropout, Flatten
from keras.constraints import MaxNorm
from keras.layers.convolutional import Conv2D, MaxPooling2D
import numpy as np
import matplotlib.pyplot as pyplot
from utils import get_labels, to_categorical, unpickle

"""
    CIFAR100:
        This is a dataset of 50,000 32x32 color training images and
        10,000 test images, labeled over 100 fine-grained classes that are
        grouped into 20 coarse-grained classes.

    Each image comes with a "fine" label (the class to which it belongs) and a "coarse" label (the superclass to which it belongs)
"""
# get list of fine and coarse labels
(fine_labels, coarse_labels) = get_labels()
# state = unpickle("imgs/cifar-100-python/train")

# images and labels are lists that match up. first index element of images matches to first index element of label
(train_images, train_labels), (test_images, test_labels) = cifar100.load_data()
# print(state["coarse_labels"][0]) cattle - 11 coarse category
# print(state["fine_labels"][0]) cattle - 19 fine category

# transforms each 1x100 row into 0s and 1s where the 1 represents in the index related to the fine grained label
# NOTE: the activation function will output a probability for correct with respect to each element. we would have to normalize test_images.
# testing_label = to_categorical(train_labels)
# print(testing_label[0])

# max_index = np.argmax(testing_label[0])
# print(fine_labels[max_index])
# train_images will access the 32 by 32 matrix representation of our first image
# img = train_images[0]

# img[0] will access the 1 by 32 vector representation of the first row of pixels in our image
# print(img[0])

# display image [hover around the first row to see that values match img[0]]
# pyplot.imshow(img)
# pyplot.show()

# FORMATTING AND NORMALIZING CIFAR-10 DATASET
# normalize labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 255 -> 1, 0 -> 0 with floats between 0 and 1 to represent 255 bit color channels
# print(train_images[0])
train_images = train_images.astype("float32") / 255.0
# print(train_images[0])
test_images = test_images.astype("float32") / 255.0


# Building the model!
# A Sequential model inputs or outputs sequences of data, as a plain stack of layers where each layer has one input tensor and one output tensor
model = Sequential()

# Conv2D - takes in input layer and outputs a tensor ! First Layer
# filters - number of output features
# kernel_size - size of the filter we want, determines output features
# recall that our input is 32x32 pixels with 3 different color channels (r,g,b)

model.add(
    Conv2D(
        filters=32,
        kernel_size=(3, 3),
        input_size=(32, 32, 3),
        padding="same",
        kernel_constraint=MaxNorm(3),
        activation="relu",
    )
)

# Max Pooling - simplifies the input along spatial dimensions by taking the max value over an input window.
model.add(MaxPooling2D(pool_size=(2, 2)))

# Will flatten the matrix of features to a stream (list) of features
model.add(Flatten())

# Creates a dense layer that is connected to the flattened tensor.
model.add(Dense(units=512, kernel_constraint=MaxNorm(3), activation="relu"))

# Prevents over-fitting. drops half the neurons
model.add(Dropout(rate=0.5))

# Have 100 output categories, use softmax when working with probabilties
model.add(Dense(units=100, activation="softmax"))
