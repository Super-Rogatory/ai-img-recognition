from keras.models import Sequential
from keras.datasets import cifar100
from keras.layers import Dense, Dropout, Flatten
from keras.constraints import maxnorm
from keras.layers.convolutional import Conv2D, MaxPooling2D
from utils import get_labels, to_categorical
import os

print()
# get list of fine and coarse labels
(fine_labels, coarse_labels) = get_labels()

# images and labels are lists that match up. first index element of images matches to first index element of label
(train_images, train_labels), (test_images, test_labels) = cifar100.load_data()

# 255 -> 1, 0 -> 0 with floats between 0 and 1 to represent 255 bit color channels
train_images = train_images.astype("float32") / 255.0
test_images = test_images.astype("float32") / 255.0

# normalize labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

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
        input_shape=(32, 32, 3),
        padding="same",
        kernel_constraint=maxnorm(3),
        activation="relu",
    )
)

# Max Pooling - simplifies the input along spatial dimensions by taking the max value over an input window.
model.add(MaxPooling2D(pool_size=(2, 2)))

# Will flatten the matrix of features to a stream (list) of features
model.add(Flatten())

# Creates a dense layer that is connected to the flattened tensor.
model.add(Dense(units=512, kernel_constraint=maxnorm(3), activation="relu"))

# Prevents over-fitting. drops half the neurons
model.add(Dropout(rate=0.5))

# Have 100 output categories, use softmax when working with probabilties
model.add(Dense(units=100, activation="softmax"))

# Configure the model for training. Adam seemed to work better than SGD (momentum algorithm)
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# Trains model for fixed number of epochs..the number of times we are iterating through the training data.
model.fit(x=train_images, y=train_labels, epochs=1, batch_size=32)

model.save(filepath=os.getcwd() + "/src/image_classifier.h5")
