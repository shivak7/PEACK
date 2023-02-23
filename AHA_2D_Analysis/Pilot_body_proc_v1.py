import os
import scipy
import numpy as np
import sys
import pickle
import time
sys.path.insert(1, 'C:\\Users\\shiva\\Dropbox\\Burke Work\\DeepMarker\\Processed Data\\PythonScripts\\PEACK_API')

from DataLoader import DataLoader
from ExtractKinematicData import ExtractKinematicData

from OP_maps import PartsMap
from TaskTimes import TaskTimes
MainDataPath = "C:\\Users\\shiva\\Dropbox\\Burke Work\\DeepMarker\\Processed Data\\AHA2D\\data0\\Body"

DL = DataLoader(MainDataPath)
PEACKData = [];

Body = PartsMap("OP_BodyMap.csv")
Hand = PartsMap("OP_HandMap.csv")
all_joints = [Body["RShoulder"], Body["RElbow"], Body["RWrist"], Body["LShoulder"], Body["LElbow"], Body["LWrist"], Body["Neck"], Body["MidHip"]])
#all_joints = np.int16([Hand["Lunate"], Hand["ThumbDP"], Hand["IndexDP"], Hand["MiddleDP"], Hand["RingDP"], Hand["LittleDP"]])

AHA_Fs = 30;

Times = TaskTimes("TimeData_I.csv")     #Load csv file containing task-segmented timestamps

for i in range(0,DL.NsubDirs):
    for j in range(0,DL.SubDirs[i].Nfiles):
        print("File: ", j)
        TimeKey = list(Times)[j]
        if TimeKey[:-1] in DL.SubDirs[0].Files[j]:      #If current timesegment file name is within kinematic data filename, they are a match!
            TaskStart = Times[TimeKey][0::2]
            TaskEnd = Times[TimeKey][1::2]
        else:
            print("Error! Missing file for: ", TimeKey, " OR Task Time Segmented Data not in sorted order!")
            raise SystemExit
        tempStruct = []
        for t in range(len(TaskStart)):
            start = TaskStart[t]*1000           # Convert time in seconds to milliseconds
            stop = TaskEnd[t]*1000
            tempdata = ExtractKinematicData(DL.getFile(i,j), all_joints, AHA_Fs, len(all_joints), smoothing_alpha = 0.1, cutoff = 3, order = 5, median_filter=0.1, trunc= [start,stop], unit_rescale=1.0, type='PEACK', filtered = False, drop_lower=True)
            tempStruct.append(tempdata)
            print("Task ", t, " Kinematic Data Extracted")
        PEACKData.append(tempStruct)



with open('AHA_2015_Initial_lhand.pkl', 'wb') as f:
    pickle.dump(PEACKData, f)
#import pdb;
#pdb.set_trace()


#Originally Used 0.1 for smoothing alpha
