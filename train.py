from datetime import time

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import TensorBoard
import numpy

dataset = numpy.loadtxt('train_transformed.csv', delimiter=',')
# Y is the second column (index 1) which is 'Survived'
Y = dataset[:,1]
# X includes all columns from the third column (index 2) onwards
X = dataset[:,2:]

model = Sequential()
# Update input_dim to reflect the new number of features in X
model.add(Dense(11, input_dim=X.shape[1], activation='relu'))
model.add(Dense(11, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
history = model.fit(X,Y, epochs=1000, batch_size=500)
loss, accuracy = model.evaluate(X,Y)
print(f"Loss. {loss}, Accuracy: {accuracy*100}")

model.save('model.hdf5')
