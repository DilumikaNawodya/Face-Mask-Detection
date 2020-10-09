# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 14:38:59 2020

@author: Dilumika
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')

X_valid = np.load('X_valid.npy')
y_valid = np.load('y_valid.npy')

X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')


from keras.models import Sequential
from keras.layers import Conv2D,Flatten,Activation,Dense,Dropout,MaxPooling2D
from keras.callbacks import ModelCheckpoint

model = Sequential()

model.add(Conv2D(200,(3,3),input_shape = X_valid.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(100,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dropout(0.2))

model.add(Dense(50,activation='relu'))
model.add(Dense(2,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()


checkpoint = ModelCheckpoint('model-{epoch:03d}.model',
                             monitor='val_loss',
                             verbose=0,
                             save_best_only=True,
                             mode='auto')

history = model.fit(X_train,y_train,epochs=20,callbacks=[checkpoint],validation_data=(X_valid,y_valid))

model.evaluate(X_test,y_test)