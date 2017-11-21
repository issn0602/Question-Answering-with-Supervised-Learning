#sentiment analysis on the IMDB dataset with Keras.
#based on the tutorial by chollet

from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb

# hyperparameter setting
max_features = 30000
maxlength = 80
batch_size = 32 #tune based on GPU memory.
print("maximum features",max_features)
print("maximum length to consider",maxlength)
print("batch size for processing",batch_size)
(x_training, y_training), (x_testing, y_testing) = imdb.load_data(num_words=max_features)
print(len(x_training), 'number of training sequences')
print(len(x_testing), 'number of testing sequences')

#padding to account for different sizes of the sequences
print('Pad sequences (samples x time) based on maxlength')
x_training = sequence.pad_sequences(x_training, maxlen=maxlength)
x_testing = sequence.pad_sequences(x_testing, maxlen=maxlength)
print('x_train shape:', x_training.shape)
print('x_test shape:', x_testing.shape)

print('Building model')
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(x_training, y_training,
          batch_size=batch_size,
          epochs=15,
          validation_data=(x_testing, y_testing))
score, acc = model.evaluate(x_testing, y_testing,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)