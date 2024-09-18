from keras import models
from keras import layers
from keras import optimizers
from keras.src.saving import load_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np

model = load_model('h5model/h5model.h5')

data = [0, 19, 195, 88, 40, 140, 38]

data = np.array(data).reshape(1, -1)

result = model.predict(data)
print(result)