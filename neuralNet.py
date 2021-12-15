import sys
from matplotlib import pyplot as plot
import numpy as np
from PIL import Image
from matplotlib import pyplot
from keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from tensorflow.keras.optimizers import SGD

# Load the dataset and get the test and train dataset
def load_dataset():
    (trainX, trainY), (testX, testY) = (inputFiles[:1500], targetValues[:1500]), (inputFiles[1501:1797], targetValues[1501:1797])
    trainX = np.array(trainX)
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return np.array(trainX).reshape(-1, 32, 32, 1), np.array(trainY), np.array(testX).reshape(-1, 32, 32, 1), np.array(testY)

# Changing the prixel values to float values 
def prep_pixels(train, test):
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	return train_norm, test_norm

# Adding layers to the model and creating the model
# 4 layered model with 2 hidden layers has been created
def define_model():
    model = Sequential()
    model.add(Conv2D(100, kernel_size=3, activation='relu', input_shape=(32,32, 1)))
    model.add(Conv2D(75, kernel_size=3, activation='relu'))
    model.add(Conv2D(50, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to plot the final graphs
def summarize_diagnostics(history):
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
	filename = sys.argv[0].split('/')[-1]
	pyplot.savefig(filename + '_plot.png')
	pyplot.close()

inputFiles = []
targetValues = []

print("Loading the files")
data = open('optdigits-orig.windep')
lines = data.readlines()
for j in range(0, 1797):
  sample = []
  for i in range(33*j, 33*j+33):
    sample1 = []
    for s in (list(lines[21 + i].strip())):
      sample1.append(int(s))
    sample.append(sample1)
  img = sample[:-1]
  inputFiles.append(np.array(img))
  targetValues.append(sample[-1])
print("Loaded all the images")
print("Spliting the images into train and test")
trainX, trainY, testX, testY = load_dataset()
print("Preparing the images for training")
trainX, testX = prep_pixels(trainX, testX)
print(np.shape(trainX), np.shape(trainY))
model = define_model()
print("Starting to the train the model")
history = model.fit(trainX, trainY, epochs=7, batch_size=10, validation_data=(testX, testY), verbose=1)
_, acc = model.evaluate(testX, testY, verbose=1)
print('> %.3f' % (acc * 100.0))
summarize_diagnostics(history)