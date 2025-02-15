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
    FuncList.append(BBT.wrist_velocity_raw)
    # FuncList.append(AHA.trunk_rotation_angle)
    # FuncList.append(AHA.trunk_displacement_distance)
    # FuncList.append(AHA.elbow_flexion_angle)                                        #Temporary! change back to append for single metric
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

def barplot(Pre, Post, idx):

    x = Pre[idx]
    y = Post[idx]

    #import pdb; pdb.set_trace()
    width = 0.35
    fig = plt.subplots(figsize =(10, 7))
    ind1 = np.arange(1) 
    ind2 = ind1 + width
    
    pre_means = np.mean(Pre,axis=1)
    post_means = np.mean(Post,axis=1)
    pre_stds = np.std(Pre,axis=1)/np.sqrt(len(Pre))
    post_stds = np.std(Post,axis=1)/np.sqrt(len(Post))
    
    ylabs = ['Peak vel (m/s)', 'Time to peak vel (ms)', 'Avg Acceleration (m/$s^2$)', 'Jerkiness (Sub-movements)']
    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.bar(ind1, pre_means[i], width, yerr = pre_stds[i])
        plt.bar(ind2, post_means[i], width, yerr = post_stds[i])
        plt.ylabel(ylabs[i])
        plt.gca().get_xaxis().set_ticks([])

    #plt.show()
    plt.savefig('PART.svg')


## Main code here

Datadir = 'Datafiles/'
#Generate_PEACK_Data(Datadir + "PART_Adults.xml")
Res = Analyze_PEACK_Data(Datadir + "PART_Adults.xml")
#res2015_pre, res2015_pre_labels = Analyze_PEACK_Data(Datadir + "AHA_Body_Params_2015_Pre.xml")

#df1 = ResultsDataFrame(res2015_pre, res2015_pre_labels, "Pre", "2015")
#df5 = ResultsDataFrame(res2015_post, res2015_post_labels, "Post", "2015")

#df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8], ignore_index=True)
#df.to_csv("AHA_2015-18_filtered.csv", index=False)
Post_vals = Res[0][0]
Pre_vals = Res[0][1]


barplot(Pre_vals, Post_vals, 0)
#import pdb; pdb.set_trace()
