import tensorflow as tf
from keras import models

model = models.load_model('h5model/h5model.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('calories_model.tflite', 'wb') as f:
    f.write(tflite_model)
