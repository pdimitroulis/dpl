import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
import numpy as np
from os import listdir
from skimage import io
from scipy.misc import imresize
import keras
from keras.optimizers import Adam
import matplotlib.pyplot as plt

img_size = 64
batch_size=64
epochs=50
test_size = 0.1
# load image

# ------------------------------------------------------------------------
# ------ There were no official comments. All comments are mine. ---------
# ------------------------------------------------------------------------

data_path='Data/' # specify data folder
labels = listdir(data_path) # take two subfolders (cats and dogs). listdir returns entries in that path.

x_cat=[]; # initialize
x_dog=[]; # initialize

cat_imgpath = listdir(data_path+'/'+labels[0]) # take cat images
dog_imgpath=listdir(data_path+'/'+labels[1]) # take dog images

# Resize cat images
for img in cat_imgpath:
    cat_img = io.imread(data_path+'/'+labels[0]+'/'+img)
    x_cat.append(imresize(cat_img, (img_size, img_size, 3)))

y_cat=np.ones(len(cat_imgpath)) # initialize (not used)

# Resize dog images
for img in dog_imgpath:
    dog_img = io.imread(data_path + '/' + labels[1] + '/' + img)
    x_dog.append(imresize(dog_img, (img_size, img_size, 3)))

# initialize
y_cat = np.zeros(len(cat_imgpath))
y_dog = np.ones(len(dog_imgpath))

x=np.asarray(x_cat+x_dog) # single array with cats and dogs resized images
y=np.append(y_cat,y_dog) # single array with zeros and ones

y = keras.utils.to_categorical(y) # Convert integers (ones and zeros) to binary

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.1, random_state=30) # split training and testing data

# ------------------- START of sequential network -------------------------------------
model = Sequential() # create sequential model

# initialize parameters for dimensions
kernel_height = 1 # best from q3
kernel_width = 3 # best from q3
num_kernels = 32
pool_width = 2
pool_stride = 1
conv_padding = 0
conv_stride = 1
hidden_units = 2

# --- Part 1 - Convolutional Layers ---
# conv_I = img_size # input dimension of conv layer
model.add(Dropout(0.1))
model.add(Conv2D(num_kernels, (kernel_height, kernel_width), input_shape=(img_size, img_size, 3))) # concolutional layer
model.add(Conv2D(num_kernels, (kernel_height, kernel_width))) # concolutional layer

# ^^^^ add more conv layers here ^^^^

model.add(Activation('sigmoid')) # activation function
model.add(MaxPooling2D(pool_size=(pool_width, pool_width))) # pooling

# --- Part 2 - Fully Connected Layers ---
model.add(Flatten()) # all in a single column
# model.add(Dense(64))
# model.add(Dropout(0.1))
# model.add(Activation('sigmoid'))
model.add(Dense(hidden_units)) # Just a linear operation (matrix x vector). Just a fully connected layer
model.add(Dense(hidden_units)) # Just a linear operation (matrix x vector). Just a fully connected layer

# ^^^^ add more dense layers here ^^^^

model.add(Activation('softmax')) # (L6)
# ------------------- END of sequential network -------------------------------------

adamop=Adam(lr=0.4) # learning rate ??? (not used !?)

model.compile(loss='poisson', optimizer='adam', metrics=['accuracy']) # configures the model for training

# train the model (for a given number of epochs)
history = model.fit(train_x, train_y,
                    validation_data=[test_x, test_y], # it gives us the val_accuracy metric
                    batch_size=batch_size,
                    epochs=epochs,
                    shuffle=True)

# Evaluation
score = model.evaluate(test_x, test_y) # Returns the loss value & metrics values for the model in test mode.

# Testing
pred=model.predict(test_x) # Generates output predictions for the input samples.

print('Test loss:', score[0])
print('Test accuracy:', score[1])
print(history.history.keys())

model.summary()
filename = __file__[:-3]
model.save(filename+'.h5')

# ----- Question 1 -----
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.grid()
plt.show()
