import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model

img_width, img_height = 224, 224
model_path = './models/model.h5'
model_weights_path = './models/weights.h5'
model = load_model(model_path)
model.load_weights(model_weights_path)

def predict(file):
  x = load_img(file, target_size=(img_width,img_height))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = model.predict(x)
  result = array[0]
  answer = np.argmax(result)
  if answer == 0:
    print("Label: fridge")
  elif answer == 1:
    print("Label: stove")
  elif answer == 2:
    print("Label: eattable")
  elif answer == 3:
    print("Label: sink")
 
  elif answer == 4:
    print("Label: kitcheb_locker ")
  elif answer == 5:
    print("Label:floor ")
  return answer 




predict("/home/bora3i/CNN-Image-Classifier/src/default_gzclient_camera(1)-2018-05-04T10_42_25.805444.jpg")
