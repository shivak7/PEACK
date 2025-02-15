import sys
sys.path.insert(1, '../PEACK_API')
#from PEACK import ProcessAsMultiEpochs, AnalyzeEpochs
from PEACK import ProcessAsSingleEpochs, AnalyzeAsSingleEpochs
from ParamDef import AHAParamsDemo, SaveParameters, LoadParameters
from PEACKMetrics import BBT
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
    ProcessAsSingleEpochs(AHA_Parameters)


def Analyze_PEACK_Data(ParamFile):

    ## Analyze body movements (Orientation: Shoulder + Trunk Movements)

    Parameters = LoadParameters(ParamFile)

    FuncList = []
    FuncList.append(BBT.wrist_velocity)
    Res, IDs = AnalyzeAsSingleEpochs(Parameters, FuncList)
    Res = np.squeeze(np.array(Res))
    #import pdb; pdb.set_trace()
    return Res, IDs
    #np.savetxt("Orientation.csv", Res, delimiter=",", fmt='%.2f')

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
Generate_PEACK_Data(Datadir + "BBT_Control_Adults_Comf.xml")
Generate_PEACK_Data(Datadir + "BBT_Control_Adults_Fast.xml")

Res, IDs = Analyze_PEACK_Data(Datadir + "BBT_Control_Adults_Comf.xml")
df1 = ResultsDataFrame(Res, IDs, "Comf")

Res, IDs = Analyze_PEACK_Data(Datadir + "BBT_Control_Adults_Fast.xml")
df2 = ResultsDataFrame(Res, IDs, "Fast")

df = pd.concat([df1, df2], ignore_index=True)
df.to_csv(Datadir + "BBT_Adult_Healthy_2023.csv", index=False)

#import pdb; pdb.set_trace()
