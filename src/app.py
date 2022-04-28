from keras.datasets import cifar10
import matplotlib.pyplot as pyplot

"""
    Recall from the source website that the CIFAR-10 dataset consists of 60000 images that are size 32 by 32.
    There are also 50000 training images and 10000 test images.
"""
# images and labels are lists that match up. first index element of images matches to first index element of label
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# train_images will access the 32 by 32 matrix representation of our first image
img = train_images[0]
# img[0] will access the 1 by 32 vector representation of the first row of pixels in our image
print(img[0])
# display image [hover around the first row to see that values match img[0]]
pyplot.imshow(img)
pyplot.show()
