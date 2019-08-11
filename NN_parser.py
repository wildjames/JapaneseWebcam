# Train a tensorflow model on the training data

import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    'grab',
    help='How many frames to grab',
    type=int
)
args = parser.parse_args()
N_grab = args.grab

CHARLIST = [
    '00', '01', '02', '03', '04', '05', '06', '07', '08', '09',
    '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
    '20', '21', '22', '23', '24', '25', '26', '27', '28', '29'
]

training_images = []
training_answer = []
with open("Training_sample/data.txt") as f:
    for line in f:
        path, value = line.strip().split(" ")

        # Get the number in the image
        value = int(value)
        value = "{:02d}".format(value)

        # Get the image data (20x20)
        image_raw = tf.io.read_file(path)
        image_tf  = tf.image.decode_image(image_raw)

        training_images.append(image_tf)
        training_answer.append(CHARLIST.index(value))

training_answer = np.asarray(training_answer)
training_images = np.asarray(training_images)

# Scale the images from 0 -> 255 to 0 -> 1
training_images = training_images / 255.0

# The above reads images as (X, Y, COLORS). Even when B/W, the COLORS are a
# list of one. Flatten that out
training_images = np.squeeze(training_images)

# Split the training set into two, for training and testing
n_images = len(training_answer)
sl = int(n_images/2)

train_data = training_images[sl:, ...]
train_keys = training_answer[sl:, ...]

test_data = training_images[:sl, ...]
test_keys = training_answer[:sl, ...]

# Plot a sample of the images
plt.figure(figsize=(4,4))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_data[i], cmap=plt.cm.binary)
    plt.xlabel(CHARLIST[train_keys[i]])
plt.show()


# Build a model
input_shape = train_data[0].shape
middle_layer = 128
output_shape = len(CHARLIST)
print("Model will have the layers:")
print("{} --> {} --> {}".format(input_shape, middle_layer, output_shape))

model = keras.Sequential([
    keras.layers.Flatten(input_shape=input_shape),
    keras.layers.Dense(middle_layer, activation=tf.nn.relu),
    keras.layers.Dense(output_shape, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# Train the model on the data
model.fit(
    train_data,
    train_keys,
    epochs=100,
    verbose=0
)

print("\n\nTESTING ON SECOND DATASET:")
# Let's test it.
test_loss, test_acc = model.evaluate(test_data, test_keys)
print('Test accuracy:', test_acc)
if test_acc != 1.0:
    print("Get more training data!")
    exit()


# The URL to this camera
# http://www.insecam.org/en/view/805350/
URL = "http://210.148.5.180:8084/?action=stream"

# Open a connection to the weird camera
cap = cv2.VideoCapture(URL)

# I need a permanent dir to hold training data for OCR
if not os.path.isdir("Training_sample"):
    os.mkdir("Training_sample")

margin = 180

with open("scraped.txt", 'w') as f:
    for _ in range(N_grab):
        ret, image = cap.read()

        # Massage into something the NN can parse
        image = image[margin:-margin, margin:-margin]
        frame = cv2.resize(image, (20,20))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, frame = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)

        img_tf = np.array([frame])
        val = model.predict(img_tf)
        index = np.argmax(val)

        # Get the answer
        num = CHARLIST[index]

        f.write("{},".format(num))

        # show to the user
        print("I think this image is {}".format(num))
        # cv2.imshow('Grabbed this', image)
        # cv2.waitKey(10)