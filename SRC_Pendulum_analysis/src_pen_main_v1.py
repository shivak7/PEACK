import sys
sys.path.insert(1, '../PEACK_API')
#from PEACK import ProcessAsMultiEpochs, AnalyzeEpochs
from PEACK import ProcessAsSingleEpochs, AnalyzeAsSingleEpochs
from ParamDef import AHAParamsDemo, SaveParameters, LoadParameters
from PEACKMetrics import Pendulum
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
    FuncList.append(Pendulum.angular_velocity)
    Res = AnalyzeAsSingleEpochs(Parameters, FuncList)
    #import pdb; pdb.set_trace()
    Res = np.squeeze(np.array(Res))
    return Res
    #np.savetxt("Orientation.csv", Res, delimiter=",", fmt='%.2f')

# def ResultsDataFrame(ResultValues, SubjectLabels, GroupLabel, YearLabel):

#     Value = []
#     Subject = []
#     Task = []
#     Measures = []#["Trunk Sway Angle", "Trunk Rotation Angle", "Trunk Sway Distance"]
#     Group = []
#     for i in range(len(ResultValues)):          #Subjects
#         for j in range(len(ResultValues[i])):       # Tasks
#                 Value.append(ResultValues[i][j][:])
#                 Task.append("Task " + str(j + 1))
#                 Group.append(GroupLabel)
#                 Subject.append(SubjectLabels[i] + '_' + YearLabel)
#     npVal = np.array(Value)
#     df_list = list(zip(Subject, npVal[:,0], npVal[:,1], npVal[:,2], npVal[:,3], Task, Group))
#     #import pdb; pdb.set_trace()
#     df = pd.DataFrame(df_list,
#                   columns=["Subject ID", "Trunk Angular Displacement", "Trunk Rotation Angle", "Trunk linear Distance", "Elbow Flexion L/R", "Task", "Group"])

#     #import pdb; pdb.set_trace()
#     return df

## Main code here

Datadir = 'Datafiles/'
Generate_PEACK_Data(Datadir + "SRC_Pen_Adults.xml")
Res = Analyze_PEACK_Data(Datadir + "SRC_Pen_Adults.xml")
#res2015_pre, res2015_pre_labels = Analyze_PEACK_Data(Datadir + "AHA_Body_Params_2015_Pre.xml")

#df1 = ResultsDataFrame(res2015_pre, res2015_pre_labels, "Pre", "2015")
#df5 = ResultsDataFrame(res2015_post, res2015_post_labels, "Post", "2015")

#df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8], ignore_index=True)
#df.to_csv("AHA_2015-18_filtered.csv", index=False)

#import pdb; pdb.set_trace()
