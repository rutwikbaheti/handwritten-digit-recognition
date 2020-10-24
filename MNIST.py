from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense
from keras.datasets import mnist
from keras.utils import np_utils as np
import numpy

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000,28, 28, 1).astype("float32")
x_test = x_test.reshape(10000,28, 28, 1).astype("float32")

x_train /= 255
x_test /= 255

y_train = np.to_categorical(y_train)
y_test = np.to_categorical(y_test)

cnn_model = Sequential()
cnn_model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation="relu"))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Conv2D(64, (3, 3), activation="relu"))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Conv2D(128, (3, 3), activation="relu"))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Flatten())
cnn_model.add(Dense(units=128, activation="relu"))
cnn_model.add(Dense(units=10, activation="softmax"))

cnn_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

cnn_model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=200, epochs=10)

scores = cnn_model.evaluate(x_test, y_test, verbose=0)
print("Error: {:.2f}%".format((1-scores[1])*100))