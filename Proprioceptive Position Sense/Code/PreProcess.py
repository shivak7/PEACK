import numpy as np
import pandas as pd

def PreProcess(Data):

  Data[Data==0]='nan'
  X = pd.DataFrame(Data);
  Y = X.interpolate(method='spline',order=5,axis=0).ffill().bfill()

  return Y.values
