import pandas as pd
import tensorflow as tf
from tensorflow import keras

from dlpp.price_data import PriceData
from configs import bc1 as cfg

def get_price_data():
  #'pdata = PriceData(cfg["ETH_MIN_OHLCV_PATH"])
  pdata = PriceData(cfg["ETH_OHLCV_PATH"])

  return pdata


def make_train_data(pdata):
  learn_data = pd.DataFrame({
    'sma5' : pdata.sma_diff("close",5),
    'sma20' : pdata.sma_diff("close",20),
    'sma50' : pdata.sma_diff("close",50),
    'sma100' : pdata.sma_diff("close",100),
    'sma200' : pdata.sma_diff("close",200),
    'sma400' : pdata.sma_diff("close",400),
    'vt_sma5' : pdata.sma_diff('volumeto', 5),
    'vt_sma50' : pdata.sma_diff('volumeto', 50),
    'vt_sma200' : pdata.sma_diff('volumeto', 200),
    'vt_sma400' : pdata.sma_diff('volumeto', 400),
    'obv' : pdata.OBV(), 
    'obvm_5' : pdata.OBVM(5),
    'obvm_50' : pdata.OBVM(50),
    'obvm_200' : pdata.OBVM(200),
    'obvm_400' : pdata.OBVM(400),
    'labels' : pdata.future_sma_bias('close',12)
  }).dropna()[20000:]

  return (learn_data, learn_data.pop('labels'))

def make_model():
  model = keras.Sequential([
    keras.layers.Dense(2000, activation=tf.nn.relu, input_shape=[len(train_data.keys())]),
    keras.layers.Dense(2000, activation=tf.nn.relu),
    keras.layers.Dense(2000, activation=tf.nn.relu),
    keras.layers.Dense(2, activation=tf.nn.softmax)
  ])

  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  model.summary()

  return model

model, train_data, train_labels = (None, None, None)

def run(save=False):
  if save:
    callbacks= [
      tf.keras.callbacks.ModelCheckpoint(cfg['CHECKPOINT_PATH'],
                                         monitor='val_acc',
                                         verbose=1,
                                         save_best_only=True,
                                         save_weights_only=True)
    ]
  else:
    callbacks=[]
    
  history = model.fit(train_data, train_labels.values, epochs=50, validation_split=0.05, callbacks=callbacks)
    
  return history

#for ipython:
# %load_ext autoreload
# %autoreload 2

#pdata = get_price_data()
train_data, train_labels = make_train_data(pdata)

model = make_model()
run(True)
