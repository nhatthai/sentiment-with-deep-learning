# Load and prepare the dataset
import numpy
import pandas as pd

from sklearn.model_selection import train_test_split
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding

# array from NumPy to convert the dataset to NumPy arrays
# one_hot to encode the words into a list of integers
# pad_sequences that will be used to pad the sentence sequences to the same length
# Sequential to initialize the neural network
# Dense to facilitate adding of layers to the neural network
# Flatten to reshape the arrays
# Embedding that will implement the embedding layer

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# Importing the dataset
dataset = pd.read_csv(
    '/opt/data/Restaurant_Reviews.tsv', delimiter='\t', quoting=3)
docs = dataset['Review']
labels = dataset['Liked']

X_train, X_test, y_train, y_test = \
    train_test_split(docs, labels, test_size=0.3)

vocab_size = 500

# one_hot to encode the words into a list of integers
X_train = [one_hot(
    d, vocab_size, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~',
    lower=True, split=' ') for d in X_train]
X_test = [one_hot(
    d, vocab_size, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~',
    lower=True, split=' ') for d in X_test]

max_length = 32
X_train = pad_sequences(X_train, maxlen=max_length, padding='pre')
X_test = pad_sequences(X_test, maxlen=max_length, padding='pre')

# create the model
model = Sequential()
model.add(Embedding(vocab_size, 8, input_length=max_length))
model.add(Conv1D(
    filters=32, kernel_size=3, padding='same',
    activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(
    loss='binary_crossentropy', optimizer='adam',
    metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=128, verbose=2)

print(model.summary())

# Final evaluation of the model
scores = model.evaluate(X_train, y_train, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))

# evaluate the model
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# check accuracy on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print('Testing Accuracy is {} '.format(accuracy*100))
