import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
import random
import tensorflow as tf
from tensorflow import keras

df = pd.read_csv('data_update.csv')

data = df['countriesAndTerritories']
df = df.loc[data == 'France']

cond = df['cases'] > 300
df = df[cond]
abs = df['dateRep']
abs = abs[::-1]
df = df['cases']

df = df[::-1]
df = df.cumsum()

y = np.array(df.iloc[:])
x = list(enumerate(y))
random.shuffle(x)
x, y = zip(*x)
x, y = np.array(x), np.array(y)
max_y = max(y)

num = len(y)//3
y1 = y[:-num]
max_y1=max(y1)
y1 = y1/max_y1
y2 = y[-num:]
max_y2=max(y2)
x1 = x[:-num]
x2 = x[-num:]

(trainX, testX, trainY, testY) = train_test_split(x1, y1,
	test_size=0.1, random_state=42)

model = tf.keras.Sequential([
     tf.keras.layers.Dense(50, input_dim=1, activation='relu'),
     tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')
  ])


model.compile(optimizer='adam',
            loss='mean_squared_error',)

model.fit(trainX, trainY, validation_data=(testX, testY), epochs=150, batch_size=2)

predictions = model.predict(x2)
predictions = predictions.reshape(28)
predictions *= max_y2


plt.scatter(x2, predictions)
plt.scatter(x2, y2)
plt.title("COVID_19")
plt.xlabel("Days from first case")
plt.ylabel("Total cases")
plt.legend(["predictions", "Reality"])


plt.show()

diff = (predictions-y2)/y2
diff *= 100
print("Mean error (in percentage) :", np.abs(diff.mean()))

#Correct prediction here with Neural network but it could be improve with more data