import sys
sys.path.insert(1, '../PEACK_API')
from PEACK import ProcessAsMultiEpochs, ProcessAsSingleEpochs, AnalyzeEpochs, AnalyzeAsSingleEpochs
from ParamDef import AHAParamsDemo, SaveParameters, LoadParameters
from PEACKMetrics import AHA
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

    AHA_Parameters = LoadParameters(ParamFile)

    FuncList = []
    FuncList.append(AHA.trunk_displacement_angle)
    FuncList.append(AHA.trunk_rotation_angle)
    FuncList.append(AHA.trunk_displacement_distance)
    FuncList.append(AHA.elbow_flexion_angle)                                       
    Res, Labels = AnalyzeAsSingleEpochs(AHA_Parameters, FuncList)
    Res = np.squeeze(np.array(Res))
    return Res, Labels
    #np.savetxt("Orientation.csv", Res, delimiter=",", fmt='%.2f')

def ResultsDataFrame(ResultValues, SubjectLabels, GroupLabel, YearLabel):

    Value = []
    Subject = []
    Task = []
    Measures = []#["Trunk Sway Angle", "Trunk Rotation Angle", "Trunk Sway Distance"]
    Group = []
    for i in range(len(ResultValues)):          #Subjects
        for j in range(len(ResultValues[i])):       # Tasks
                Value.append(ResultValues[i][j][:])
                Task.append("Task " + str(j + 1))
                Group.append(GroupLabel)
                Subject.append(SubjectLabels[i][j] + '_' + YearLabel)
    npVal = np.array(Value)
    df_list = list(zip(Subject, npVal[:,0], npVal[:,1], npVal[:,2], npVal[:,3], npVal[:,4], Task, Group))
    #import pdb; pdb.set_trace()
    df = pd.DataFrame(df_list,
                  columns=["Subject ID", "Trunk Angular Displacement", "Trunk Rotation Angle", "Trunk linear Distance", "Elbow Flexion L", "Elbow Flexion R","Task", "Group"])

    #import pdb; pdb.set_trace()
    return df

## Main code here

Datadir = 'Datafiles/'
#Generate_PEACK_Data(Datadir + "TD_Body_Params.xml")
resTD, resTD_labels = Analyze_PEACK_Data(Datadir + "TD_Body_Params.xml")
import pdb; pdb.set_trace()
df1 = ResultsDataFrame(resTD, resTD_labels, "TD", "2023")
df1.to_csv(Datadir + "AHA_TD_PEACK.csv", index=False)


#import pdb; pdb.set_trace()
