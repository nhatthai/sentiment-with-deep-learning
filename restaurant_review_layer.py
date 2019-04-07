# Load and prepare the dataset
import pandas as pd

# Cleaning the texts
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

# fix random seed for reproducibility
seed = 7
# numpy.random.seed(seed)
top_words = 5000
max_words = 250

# Importing the dataset
dataset = pd.read_csv(
    'Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

nltk.download('stopwords')

corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if word not in set(
        stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# prepare text data
# “Term Frequency – Inverse Document
cv = TfidfVectorizer()
X = cv.fit_transform(corpus).toarray()
Y = dataset.iloc[:, 1].values
X = sequence.pad_sequences(X, maxlen=max_words)

# create the model
model = Sequential()
model.add(Dense(
    12, input_dim=max_words, kernel_initializer='uniform', activation='relu'))
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

model.compile(
    loss='binary_crossentropy', optimizer='adam',
    metrics=['accuracy'])
model.fit(X, Y, epochs=2, batch_size=128, verbose=2)

# Final evaluation of the model
scores = model.evaluate(X, Y, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# evaluate the model
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
