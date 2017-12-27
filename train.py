from datetime import time

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import TensorBoard
import numpy

dataset = numpy.loadtxt('train_transformed.csv', delimiter=',')
X = dataset[:,2:5]
Y = dataset[:,1]

model = Sequential()
model.add(Dense(7, input_dim=3, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
history = model.fit(X,Y, epochs=1000, batch_size=500)
loss, accuracy = model.evaluate(X,Y)
print(f"Loss. {loss}, Accuracy: {accuracy*100}")

model.save('model.hdf5')
