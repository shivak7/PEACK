import numpy as np
import pandas as pd

def TimeFormat(V):  #Convert notation Min.Sec => Seconds

    mins = np.int16(V)
    secs = (V-mins)*100
    return secs + mins*60

def TaskTimes(fn):

    temp = pd.read_csv(fn,header=None)

    Nrows = len(temp.index)
    Map = {}

    for i in range(Nrows):
         Vrow = temp.iloc[i]

         Times = np.double(Vrow.values[1:])
         Times = TimeFormat(Times)
         Key = str(Vrow.values[:1])
         Key = Key.replace(" ", "")
         Key = Key[2:-2]
         Map[Key] = Times
         if(len(Times)%2>0):
             print("Error! Number of Start and Stop times in table " + Key + " in file " + fn + " are unequal.")
             raise SystemExit
    return Map
