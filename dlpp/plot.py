import matplotlib as plt
import pandas as pd


def accuracy(history):
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
