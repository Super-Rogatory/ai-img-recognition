from keras.models import load_model
from keras.datasets import cifar100
from utils import get_labels, to_categorical, unpickle
from random import randint
import numpy as np
import os
import matplotlib.pyplot as plt

"""
    CIFAR100:
        This is a dataset of 50,000 32x32 color training images and
        10,000 test images, labeled over 100 fine-grained classes that are
        grouped into 20 coarse-grained classes.

    Each image comes with a "fine" label (the class to which it belongs) and a "coarse" label (the superclass to which it belongs)
"""
# get list of fine and coarse labels
(fine_labels, coarse_labels) = get_labels()

# all answers reside here
state = unpickle("imgs/cifar-100-python/test")

# images and labels are lists that match up. first index element of images matches to first index element of label
(train_images, _), (test_images, test_labels) = cifar100.load_data()

# 255 -> 1, 0 -> 0 with floats between 0 and 1 to represent 255 bit color channels
test_images = test_images.astype("float32") / 255.0

# normalize labels
test_labels = to_categorical(test_labels)

# load model
model = load_model(filepath=os.getcwd() + "/ml/image_classifier.h5")

# evaluate returns a list with loss and accuracy
loss, accuracy = model.evaluate(x=test_images, y=test_labels)

print("Test loss: ", loss)
print("Test accuracy: ", accuracy)

# All predictions for every image!
prediction = model.predict(test_images)

# Show random five predictions versus their correct answers.
for i in range(0, 5):
    # recall there are 50000 training images and 10000 test images
    rand_int = randint(0, 9999)
    # Strip the first prediction vector from matrix, and return the index with highest probability value
    guess_index = np.argmax(prediction[rand_int])
    answer_index = state["fine_labels"][rand_int]
    # Use state to get the index of correct answer
    print(f"IClass guesses: {fine_labels[guess_index]}")
    print(f"The correct answer is: {fine_labels[answer_index]}")
    plt.imshow(test_images[rand_int])
    plt.show()
