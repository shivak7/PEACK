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
         StrTimes = Vrow.values[1:].tolist()
         if isinstance(StrTimes[0], str):
            for j in range(len(StrTimes)):
                StrTimes[j] = StrTimes[j].replace(":",".")         
         Times = np.double(StrTimes)
         Times = TimeFormat(Times)
         Key = str(Vrow.values[:1])
         Key = Key.replace(" ", "")
         Key = Key[2:-2]
         Map[Key] = Times
         #import pdb; pdb.set_trace()
         if(len(Times)%2>0):
             print("Error! Number of Start and Stop times in table " + Key + " in file " + fn + " are unequal.")
             raise SystemExit
    return Map
