from keras.models import load_model
import pandas as pd
import numpy

dataset = numpy.loadtxt('test_transformed.csv', delimiter=',')
X = dataset[:,1:7]

model = load_model('model.hdf5')

probabilities= model.predict(X)
predictions = [float(numpy.round(x)) for x in probabilities]
out = pd.DataFrame({'PassengerId': [int(x) for x in dataset[:,0]],
                    'Survived': [int(x) for x in predictions]})
out.to_csv('result.csv', index=False)
