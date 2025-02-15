import sys
sys.path.insert(1, '../PEACK_API')
#from PEACK import ProcessAsMultiEpochs, AnalyzeEpochs
from PEACK import ProcessAsSingleEpochs, AnalyzeAsSingleEpochs
import PEACK_Filters as PF
from ParamDef import AHAParamsDemo, SaveParameters, LoadParameters
from PEACKMetrics import BBT
from PEACKMetrics import reaching
import numpy as np
import pandas as pd
from readEEG import np_extract_emg
from matplotlib import pyplot as plt
import pickle

def Generate_PEACK_Data(ParamFile):
    ##Process Body Kinematics
    BBT_Parameters = LoadParameters(ParamFile)
    ProcessAsSingleEpochs(BBT_Parameters)

def Analyze_PEACK_Data(ParamFile):

    BBT_Parameters = LoadParameters(ParamFile)
    FuncList = []
    FuncList.append(BBT.wrist_velocity)
    Res, Labels = AnalyzeAsSingleEpochs(BBT_Parameters, FuncList)

    return Res, Labels

#filename = '20241121111722.EEG'
EMG_DataDir = '../../Data/BBT_EMG/EMG Data/'
EMG_filename = EMG_DataDir + '20241217113859.EEG'
Datadir = 'Datafiles/'


EMG = np_extract_emg(EMG_filename, isMEP=False)

EDC_data = EMG.data[:,0] - EMG.data[:,1]
FPL_data = EMG.data[:,2] - EMG.data[:,3]
FDI_data = EMG.data[:,4] - EMG.data[:,5]
FDP_data = EMG.data[:,6] - EMG.data[:,7]
ECR_data = EMG.data[:,8] - EMG.data[:,9]
BIC_data = EMG.data[:,10] - EMG.data[:,11]

sig1 = BIC_data
sig2 = FDI_data

sig1_filt = PF.butter_highpass_filter(sig1, 5, EMG.fa, order=3)
sig2_filt = PF.butter_highpass_filter(sig2, 5, EMG.fa, order=3)

#Generate_PEACK_Data(Datadir + "BBT_EMG_Fast.xml")

Param = LoadParameters(Datadir + "BBT_EMG_Fast.xml")
with open(Param.OutFile, 'rb') as f:
    TempData = pickle.load(f)
#res, labels = Analyze_PEACK_Data(Datadir + "BBT_EMG_Fast.xml")

elb_angle = reaching.elbow_angle(TempData[0][0], mode='raw', side='L')
time_k = TempData[0][0].time
t_sig = np.linspace(0,len(sig1)/1024,len(sig1))

plt.plot(time_k, (elb_angle - np.mean(elb_angle))/np.std(elb_angle))
plt.plot(t_sig, (sig1 - np.mean(sig1))/np.std(sig1))
plt.show()

import pdb; pdb.set_trace()