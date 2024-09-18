# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from keras import models
from keras import layers
from keras import optimizers
from keras.src.saving import load_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd

df = pd.read_csv('exercise.csv')
df['Gender'] = df['Gender'].apply(lambda n: 0 if n == 'male' else 1)

x = pd.get_dummies(df.drop(['User_ID', 'Calories'], axis=1))
y = df['Calories']

print(df['Gender'].head())

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
x_train.head()
y_train.head()

model = models.Sequential()
model.add(layers.Dense(units=64, activation='relu', input_dim=len(x_train.columns)))
model.add(layers.Dense(units=128, activation='relu'))
model.add(layers.Dense(units=256, activation='linear'))
model.add(layers.Dense(units=1))

optimizer = optimizers.SGD(learning_rate=0.00001)
model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

model.fit(x_train, y_train, epochs=400, batch_size=64)

y_pred = model.predict(x_test)
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)

model.export('tfmodel')

# hdf5
# model.save('h5model/h5model.h5')


