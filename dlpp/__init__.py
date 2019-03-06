import pandas as pd
import numpy as np
import tensorflow as tf
import ta

from tensorflow import keras
from matplotlib import pyplot as plt

def import_data():
  print("Importing data")
  return pd.read_json("data/eth_hourly_full.json").transpose()


def future_sma_bias(s,l):
  sma = s.rolling(window = l).mean().shift(-l)
  return (s-sma).dropna().map(lambda x: 1 if x>=0 else 0)
  

def std_normalize(s):
  return (s - s.mean()) / s.std()

def min_max_normalize(s):
  return (s - s.min()) / (s.max() - s.min())

def norm_sma_diff(s,l):
  sma = s.rolling(window = l).mean()
  dif = s-sma
  return std_normalize(dif)

def split_classification():
  model = keras.Sequential([
    keras.layers.Dense(200, activation=tf.nn.relu, input_shape=[len(train_data.keys())]),
    keras.layers.Dense(200, activation=tf.nn.relu),
    keras.layers.Dense(200, activation=tf.nn.relu),
    keras.layers.Dense(2, activation=tf.nn.softmax)
  ])

  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  return model

def binary_classification():
  model = keras.Sequential([
    keras.layers.Dense(150, activation=tf.nn.relu, input_shape=[len(train_data.keys())]),
    keras.layers.Dense(150, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
  ])

  model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])

  return model

def regression():
  model = keras.Sequential([
    keras.layers.Dense(150, activation=tf.nn.relu, input_shape=[len(train_data.keys())]),
    keras.layers.Dense(150, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
  ])
  


#df = import_data()

learn_data = pd.DataFrame({
  'sma5' : norm_sma_diff(df['close'], 5),
  'sma20' : norm_sma_diff(df['close'], 20),
  'sma50' : norm_sma_diff(df['close'], 50),
  'sma100' : norm_sma_diff(df['close'], 100),
  'sma200' : norm_sma_diff(df['close'], 200),
  'vt_sma5' : norm_sma_diff(df['volumeto'], 5),
  'vt_sma50' : norm_sma_diff(df['volumeto'], 50),
  'vt_sma200' : norm_sma_diff(df['volumeto'], 200),
  'obv' : std_normalize(ta.on_balance_volume(df["close"],df["volumeto"])),
  'obvm_5' : std_normalize(ta.on_balance_volume_mean(df["close"],df["volumeto"],5)),
  'obvm_50' : std_normalize(ta.on_balance_volume_mean(df["close"],df["volumeto"],50)),
  'obvm_200' : std_normalize(ta.on_balance_volume_mean(df["close"],df["volumeto"],200)),
  'labels' : future_sma_bias(df['close'],24)
}).dropna()

train_data = learn_data.sample(frac=0.95, random_state=0)
train_labels = train_data.pop('labels')

test_data = learn_data.drop(train_data.index)
test_labels = test_data.pop('labels')

model = split_classification()
#model = binary_classification()

model.summary()

history = model.fit(train_data, train_labels.values, epochs=50, validation_split=0.2)

def plot_accuracy(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.plot(hist['epoch'], hist['acc'],
           label='Train acc')
  plt.plot(hist['epoch'], hist['val_acc'],
           label = 'Val acc')
  plt.ylim([0,1])
  plt.legend()
  
  plt.legend()
  plt.show()



# f1, ax = plt.subplots(figsize = (10,5))
# ax.plot(ohclv.index, ohclv['close'])

# ax.plot(df.index, df['SMA5'], color = 'green', label = 'SMA5')
# ax.plot(df.index, df['SMA20'], color = 'red', label = 'SMA20')

# plt.xlabel('Date')  
# plt.ylabel('Close Price')

# plt.show()

