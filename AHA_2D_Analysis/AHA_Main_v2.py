import sys
sys.path.insert(1, '../PEACK_API')
from PEACK import ProcessAsMultiEpochs, AnalyzeEpochs
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
    ProcessAsMultiEpochs(AHA_Parameters)

    #import pdb; pdb.set_trace()
    ##Process Hand Kinematics

    # AHA_Parameters = LoadParameters("AHA_LHand_Params.xml")
    # ProcessAsSingleEpochs(AHA_Parameters)
    #
    # AHA_Parameters = LoadParameters("AHA_RHand_Params.xml")
    # ProcessAsSingleEpochs(AHA_Parameters)


def Analyze_PEACK_Data(ParamFile):

    ## Analyze body movements (Orientation: Shoulder + Trunk Movements)

    AHA_Parameters = LoadParameters(ParamFile)

    FuncList = []
    FuncList.append(AHA.trunk_displacement_angle)
    FuncList.append(AHA.trunk_rotation_angle)
    FuncList.append(AHA.trunk_displacement_distance)
    FuncList.append(AHA.elbow_flexion_angle)
    FuncList.append(AHA.hand_coordination)
    #FuncList.append(AHA.reach_stats_v2)
    Res, Labels = AnalyzeEpochs(AHA_Parameters, FuncList)
    #import pdb; pdb.set_trace()
    Res = np.squeeze(np.array(Res))
    return Res, Labels
    

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
                Subject.append(SubjectLabels[i] + '_' + YearLabel)
    npVal = np.array(Value)
    df_list = list(zip(Subject, npVal[:,0], npVal[:,1], npVal[:,2], npVal[:,3], npVal[:,4], npVal[:,5], Task, Group))
    #import pdb; pdb.set_trace()
    df = pd.DataFrame(df_list,
                  columns=["Subject ID", "Trunk Angular Displacement", "Trunk Rotation Angle", "Trunk linear Distance", "Elbow Flexion L", "Elbow Flexion R", "Hand Coordination", "Task", "Group"])

    #import pdb; pdb.set_trace()
    return df

## Main code here

Datadir = 'Datafiles/'
# Generate_PEACK_Data(Datadir + "AHA_Body_Params_2015_Pre.xml")
# Generate_PEACK_Data(Datadir + "AHA_Body_Params_2016_Pre.xml")
# Generate_PEACK_Data(Datadir + "AHA_Body_Params_2017_Pre.xml")
# Generate_PEACK_Data(Datadir + "AHA_Body_Params_2018_Pre.xml")
# Generate_PEACK_Data(Datadir + "AHA_Body_Params_2015_Post.xml")
# Generate_PEACK_Data(Datadir + "AHA_Body_Params_2016_Post.xml")
# Generate_PEACK_Data(Datadir + "AHA_Body_Params_2017_Post.xml")
# Generate_PEACK_Data(Datadir + "AHA_Body_Params_2018_Post.xml")

res2015_pre, res2015_pre_labels = Analyze_PEACK_Data(Datadir + "AHA_Body_Params_2015_Pre.xml")
res2016_pre, res2016_pre_labels = Analyze_PEACK_Data(Datadir + "AHA_Body_Params_2016_Pre.xml")
res2017_pre, res2017_pre_labels = Analyze_PEACK_Data(Datadir + "AHA_Body_Params_2017_Pre.xml")
res2018_pre, res2018_pre_labels = Analyze_PEACK_Data(Datadir + "AHA_Body_Params_2018_Pre.xml")

res2015_post, res2015_post_labels = Analyze_PEACK_Data(Datadir + "AHA_Body_Params_2015_Post.xml")
res2016_post, res2016_post_labels = Analyze_PEACK_Data(Datadir + "AHA_Body_Params_2016_Post.xml")
res2017_post, res2017_post_labels = Analyze_PEACK_Data(Datadir + "AHA_Body_Params_2017_Post.xml")
res2018_post, res2018_post_labels = Analyze_PEACK_Data(Datadir + "AHA_Body_Params_2018_Post.xml")

df1 = ResultsDataFrame(res2015_pre, res2015_pre_labels, "Pre", "2015")
df2 = ResultsDataFrame(res2016_pre, res2016_pre_labels, "Pre", "2016")
df3 = ResultsDataFrame(res2017_pre, res2017_pre_labels, "Pre", "2017")
df4 = ResultsDataFrame(res2018_pre, res2018_pre_labels,"Pre", "2018")

df5 = ResultsDataFrame(res2015_post, res2015_post_labels, "Post", "2015")
df6 = ResultsDataFrame(res2016_post, res2016_post_labels, "Post", "2016")
df7 = ResultsDataFrame(res2017_post, res2017_post_labels, "Post", "2017")
df8 = ResultsDataFrame(res2018_post, res2018_post_labels, "Post", "2018")

df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8], ignore_index=True)
df.to_csv(Datadir + "AHA_2015-18_filtered_10132024_Coordination.csv", index=False)

#import pdb; pdb.set_trace()
