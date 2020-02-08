# Name: Pantelis Dimitroulis

import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, RNN, LSTM
from sklearn.model_selection import train_test_split
import numpy as np
from os import listdir
from skimage import io
from scipy.misc import imresize
import keras
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

# initial parameters
batch_size=64
epochs=50
test_size = 0.1

# ------------ sample code (START) --------------
# ---------------- Prepare dataset ------------------
df = pd.read_csv('train.csv',encoding='latin-1') # read csv file / read data
df.drop(['ItemID'], axis=1, inplace=True) # erase 1st column i.e. ItemID
label=list(df.Sentiment) # take the labels: Sentiment
text=list(df.SentimentText) # take the text of each label: SentimentText
tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True,split=" ") # create tokenizer / remove all these symbol characters
tokenizer.fit_on_texts(text) # fit the tokenizer to the text
vocab = tokenizer.word_index # Take vocabulary: A dictionary of words and their uniquely assigned integers.
X_train, X_test, y_train, y_test = train_test_split(text, label, test_size=0.1,random_state=42) # split the dataset
X_train_word_ids = tokenizer.texts_to_sequences(X_train)
X_test_word_ids = tokenizer.texts_to_sequences(X_test)
x_train = pad_sequences(X_train_word_ids, maxlen=50)
x_test= pad_sequences(X_test_word_ids, maxlen=50)
# ------------ hw3 sample code (END) --------------

# ------------------- START of sequential network -------------------------------------
model = Sequential() # create sequential model

# initialize parameters for dimensions
kernel_width = 3
kernel_height = 3
num_kernels = 32
pool_width = 2
pool_stride = 1
conv_padding = 0
conv_stride = 1
hidden_units = 2

x_train = np.expand_dims(x_train, axis=2) # reshape (89990, 50) to (89990, 50, 1)
x_test = np.expand_dims(x_test, axis=2) # reshape (89990, 50) to (89990, 50, 1)
y_train = keras.utils.to_categorical(y_train) # Convert integers (ones and zeros) to binary
y_test = keras.utils.to_categorical(y_test) # Convert integers (ones and zeros) to binary

print(x_train.shape)
# print(x_train)
print(len(y_train))
# print(y_train)

data_height = x_train.shape[0]
data_width = x_train.shape[1]
timesteps = data_width
data_dim = 1

model.add(LSTM(32, return_sequences=True, input_shape=(timesteps, data_dim)))
model.add(Activation('sigmoid')) # activation function
model.add(MaxPooling1D(pool_size=pool_width)) # pooling

model.add(Flatten()) # all in a single column
# model.add(Dense(64))
# model.add(Dropout(0.1))
# model.add(Activation('sigmoid'))
model.add(Dense(hidden_units)) # Just a linear operation (matrix x vector) and an activation function. Just a fully connected layer
model.add(Activation('softmax'))
# ------------------- END of sequential network -------------------------------------

adamop=Adam(lr=0.1) # learning rate

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # configures the model for training

# train the model (for a given number of epochs)
history = model.fit(x_train, y_train, validation_data=[x_test, y_test], batch_size=batch_size, epochs=epochs, shuffle=True) # it gives us the val_accuracy metric

# Evaluation
score = model.evaluate(x_test, y_test) # Returns the loss value & metrics values for the model in test mode.

# Testing
pred=model.predict(x_test) # Generates output predictions for the input samples.

print('Test loss:', score[0])
print('Test accuracy:', score[1])
print(history.history.keys())

model.summary() # print summary of the model

# ------- Plot accuracy -----------
# summarize history for accuracy
plt.figure(0)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.grid()
plt.show()

# summarize history for accuracy
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.grid()
plt.show()
