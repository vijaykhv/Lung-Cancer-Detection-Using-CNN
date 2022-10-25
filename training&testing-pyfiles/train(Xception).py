import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense
from tensorflow.keras.applications import Xception
import pickle
from tensorflow import keras 

train_data= "brain_tumor_dataset"
# test_data="brain_tumor_dataset"
batch_size=32
target_size=(125,125)

train = ImageDataGenerator(
    rescale=1/255.0,
    validation_split=0.40)
    
test = ImageDataGenerator(rescale=1/255.0)

train_generator = train.flow_from_directory(
    directory=train_data,
    target_size=target_size,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
    subset='training')

valid_generator = train.flow_from_directory(
    directory=train_data,
    target_size=target_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset='validation',
    shuffle=True)

# test_generator = test.flow_from_directory(
#     directory=test_data,
#     target_size=target_size,
#     batch_size=1)

# print(train_generator.classes)
model = Sequential()

model.add(Xception(include_top=False,
    weights=None,
    input_shape=(125,125,3)))
model.add(Flatten())
model.add(Dense(300, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=['accuracy'])


hist = model.fit(train_generator,validation_data = valid_generator,epochs=4)

model_json = model.to_json()
with open("model/Xception/model.json", "w") as json_file:
    json_file.write(model_json)
    f = open('model/Xception/history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()
