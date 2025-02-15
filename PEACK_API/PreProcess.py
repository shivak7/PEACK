'''
    Copyright 2022 Shivakeshavan Ratnadurai-Giridharan

    A class for pre-processing and filling gaps in PEACK / VICON kinematic data.
    This code runs cubic spline interpolation
'''
import numpy as np
import pandas as pd

def PreProcess(Data, type='3D', drop_cols = 0, do_interp = True, interp_limit=100, zeros_as_nan = False, debug = False):

  
  if zeros_as_nan:
    Data[Data==0]=np.nan
  col_step_size = 3
  X = pd.DataFrame(Data)
  if(drop_cols > 0):        #Drop first N columnns
      X = X.drop(X.columns[:drop_cols], axis=1)    # For now deleting most of the Non-upperlimb joints
  if(drop_cols < 0):        #Drop last N columns
      X = X.drop(X.columns[-drop_cols:], axis=1)
  if(type=='2D'):
      dc = np.arange(2,len(X.columns),3)
      X = X.drop(X.columns[dc], axis=1)
      col_step_size = 2



  cols = X.columns[X.isna().any()]                  # Find columns that have Nan's in them
  dc = []                                           # Columns to drop
  #import pdb; pdb.set_trace()
  for i in range(len(cols)):
      nan_ratio = np.sum(X[cols[i]].isna()) / len(X)              # For each column find the ratio of number of Nan's to the column size (number of rows).
      if(nan_ratio > 0.5):                                       # If more than 33% of a column is Nan's, mark column to be dropped.
          dc.append(cols[i])

  if(len(dc)>0):
      dc = np.array(dc,dtype='int64')
      #X = X.drop(X.columns[dc], axis=1)                           # Don't actually drop it! Just mark for removal in joint_map dict. This keeps joint indexing consistent.
      dc = np.array(dc[0::col_step_size]/col_step_size, dtype='int64')                                        #Return preprocessed data AND also the joint indices that were dropped due to insufficient data

  # if debug == True:
  #import pdb; pdb.set_trace()

  #X = X.fillna(0)
  if(do_interp == True):
      Y = X.interpolate(method='piecewise_polynomial',order=3,axis=0, limit = int(interp_limit), limit_direction='both') # limit_direction='both'
      Y = Y.ffill()
      Y = Y.bfill()    
  else:
      Y = X

  
  #import pdb; pdb.set_trace()
  #Y = Y.fillna(0)    
  return Y.values, dc
