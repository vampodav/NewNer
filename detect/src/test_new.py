import numpy as np
from keras.datasets import mnist
from keras.models import Sequential,model_from_json
from keras.layers import Dense
from keras.utils import np_utils
import cv2






color_image = cv2.imread('00001.ppm')

import matplotlib.pyplot as plt


#loss: 0.4275 - acc: 0.8862 - val_loss: 0.3359 - val_acc: 0.9091
#loss: 0.4881 - acc: 0.8502 - val_loss: 0.3277 - val_acc: 0.9091


json_file = open("midel.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('weight.h5')


loaded_model.compile(loss = "sparse_categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])




resized_image = cv2.resize(color_image, (40, 40))

test_vector = np.reshape(resized_image,(1,40,40,3))
print(test_vector.shape)


prediction = loaded_model.predict(test_vector)
print(prediction)
