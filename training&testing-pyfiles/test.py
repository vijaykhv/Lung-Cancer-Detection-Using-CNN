from unittest import result
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import pickle
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import load_img,img_to_array
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img_path = 'data/test/0002.jpeg'
img = image.load_img(img_path, target_size=(125, 125))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)

with open('model/Xception/model.json', "r") as json_file:
    loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)


print(model.summary())
f = open('model/Xception/Xceptionhistory.pckl', 'rb')
data = pickle.load(f)
f.close()


predicts = model.predict(x)


cls = np.argmax(predicts)

print(cls)

#-----------------------------------------------------------------------------------------------

# def output(location):

#     model = load_model('model/ResNet50(200ep)/model.h5')

#     lab = ['Adinocarcinoma','Benign','Squamous Cell Carcinoma']

#     img = load_img(location,target_size=(125,125))
#     img = img_to_array(img)
#     img=img/255
#     img=np.expand_dims(img,[0])
#     answer=model.predict(img)
#     y_class = answer.argmax(axis=-1)
#     y=" ".join(str(x) for x in y_class)
#     y=int(y)
#     res = lab[y]
#     return res

# img = 'dataset/Test/lung_squamous_cell_carcinoma/lungscc4001.jpeg'
# result = output(img)
# print(result)


# mat = mpimg.imread(img)
# imgplot = plt.imshow(mat)
# plt.show()