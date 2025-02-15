import sys
sys.path.insert(1, '../PEACK_API')
#from PEACK import ProcessAsMultiEpochs, AnalyzeEpochs
#from PEACK import ProcessAsSingleEpochs, AnalyzeAsSingleEpochs
import PEACK
import VICON
from ParamDef import LoadParameters
from PEACKMetrics import reaching
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# P1 = AHAParamsDemo()
# SaveParameters(P1, "test.xml")


def Generate_PEACK_Data(ParamFile):
    ##Process Body Kinematics
    AHA_Parameters = LoadParameters(ParamFile)
    PEACK.ProcessAsSingleEpochs(AHA_Parameters)


def Generate_VICON_Data(ParamFile):
    ##Process Body Kinematics
    AHA_Parameters = LoadParameters(ParamFile)
    VICON.ProcessAsSingleEpochs(AHA_Parameters)


def Analyze_PEACK_Data(ParamFile):

    Parameters = LoadParameters(ParamFile)
    FuncList = []
    FuncList.append(reaching.metrics)
    Res, IDs = PEACK.AnalyzeAsSingleEpochs(Parameters, FuncList)
    Res = np.squeeze(np.array(Res))
    return Res, IDs

def Analyze_VICON_Data(ParamFile):

    Parameters = LoadParameters(ParamFile)
    FuncList = []
    FuncList.append(reaching.metrics)
    Res, IDs = VICON.AnalyzeAsSingleEpochs(Parameters, FuncList)
    #Res = np.squeeze(np.array(Res))
    return Res, IDs


def ResultsDataFrame(ResultValues, SubjectLabels, GroupLabel, YearLabel=''):

    Value = []
    Subject = []
    Task = []
    Tasks = ['Dominant', 'Non-dominant']
    Measures = []#["Trunk Sway Angle", "Trunk Rotation Angle", "Trunk Sway Distance"]
    Group = []
    for i in range(len(ResultValues)):          # Tasks
        for j in range(len(ResultValues[i])):       #Subjects
                Value.append(ResultValues[i][j][:])
                Task.append(Tasks[i])
                Group.append(GroupLabel)
                Subject.append(SubjectLabels[i][j])# + '_' + YearLabel)
    npVal = np.array(Value)
    df_list = list(zip(Subject, npVal[:,0], npVal[:,1], npVal[:,2], npVal[:,3], Task, Group))
    df = pd.DataFrame(df_list, columns=["Subject ID", "Peak Velocity", "Time to Peak Velocity", "Peak acceleration", "Smoothness", "Handedness", "Task Speed"])
    #import pdb; pdb.set_trace()
    return df

## Main code here

Datadir = 'Datafiles/'
#Generate_PEACK_Data(Datadir + "PEACK_Adults_reaching.xml")
#Generate_VICON_Data(Datadir + "VICON_Adults_reaching.xml")

Res, IDs = Analyze_PEACK_Data(Datadir + "PEACK_Adults_reaching.xml")
#Res, IDs = Analyze_VICON_Data(Datadir + "VICON_Adults_reaching.xml")
# df1 = ResultsDataFrame(Res, IDs, "Comf")

# Res, IDs = Analyze_PEACK_Data(Datadir + "BBT_Control_Adults_Fast.xml")
# df2 = ResultsDataFrame(Res, IDs, "Fast")

# df = pd.concat([df1, df2], ignore_index=True)
# df.to_csv(Datadir + "BBT_Adult_Healthy_2023.csv", index=False)

import pdb; pdb.set_trace()
