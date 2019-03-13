import pandas as pd
import ta
from pathlib import Path

class PriceData:
  def __init__(self, import_path=None):
    if(import_path != None):
      self.import_ohclv(import_path)
    else:
      self.ohclv = pd.DataFrame()

  #TODO allow loading start/end segments  
  def import_ohclv(self,path):
    print("Importing " + path)
    self.ohclv = pd.read_json(path).transpose()

  def future_sma_bias(self,col,length):
    sma = self.ohclv[col].rolling(window = length).mean().shift(-length)
    return (self.ohclv[col]-sma).dropna().map(lambda x: 1 if x>=0 else 0)
    
  def sma_diff(self,col,length):
    sma = self.ohclv[col].rolling(window = length).mean()
    dif = (self.ohclv[col]-sma) / (sma+1)
    return std_normalize(dif)

  def OBV(self):
    return std_normalize(ta.on_balance_volume(self.ohclv["close"],self.ohclv["volumeto"]))

  def OBVM(self,length):
    return std_normalize(ta.on_balance_volume_mean(self.ohclv["close"],self.ohclv["volumeto"],length))


def std_normalize(s):
  return (s - s.mean()) / s.std()

def min_max_normalize(s):
  return (s - s.min()) / (s.max() - s.min())
