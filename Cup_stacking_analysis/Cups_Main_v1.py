import sys
sys.path.insert(1, '../PEACK_API')
from PEACK import ProcessAsMultiEpochs, AnalyzeEpochs
from ParamDef import AHAParamsDemo, SaveParameters, LoadParameters
from PEACKMetrics import AHA
from PEACKMetrics import CUPS
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
    ArgList = []
    FuncList.append(CUPS.wrist_segmented_metrics); ArgList.append({'mode':'statistic', 'side':'L'}) 
    #FuncList.append(CUPS.trunk_displacement_angle)
    #FuncList.append(AHA.trunk_rotation_angle)
    #FuncList.append(AHA.trunk_displacement_distance)
    #FuncList.append(AHA.elbow_flexion_angle)        
    #FuncList.append(CUPS.wrist_velocity)
    #FuncList.append(CUPS.hand_coordination)                                #Temporary! change back to append for single metric
    Res, Labels = AnalyzeEpochs(AHA_Parameters, FuncList, ArgList)
    #import pdb; pdb.set_trace()
    Res = np.squeeze(np.array(Res))
    return Res, Labels
    

def ResultsDataFrame(ResultValues, SubjectLabels, GroupLabel, YearLabel):

    Value = []
    Subject = []
    Task = []
    Measures = []#["Trunk Sway Angle", "Trunk Rotation Angle", "Trunk Sway Distance"]
    Group = []

    sz = ResultValues.shape
    ResultValues = ResultValues.reshape(sz[0],1,sz[1])

    for i in range(len(ResultValues)):          #Subjects
        for j in range(len(ResultValues[i])):       # Tasks
                Value.append(ResultValues[i][j][:])
                Task.append("Task " + str(j + 1))
                Group.append(GroupLabel)
                Subject.append(SubjectLabels[i] + '_' + YearLabel)
    npVal = np.array(Value)
    df_list = list(zip(Subject, npVal[:,0], npVal[:,1], npVal[:,2], npVal[:,3], npVal[:,4], npVal[:,5], npVal[:,6], npVal[:,7], Task, Group))
    #import pdb; pdb.set_trace()
    df = pd.DataFrame(df_list,
                  columns=["Subject ID", "Trunk Angular Displacement", "Trunk Rotation Angle", "Trunk linear Distance", "Elbow Flexion L", "Elbow Flexion R", "L Wrist Velocity", "R Wrist Velocity", "Hand Coordination %", "Task", "Group"])

    #import pdb; pdb.set_trace()
    return df

## Main code here
paramfiles = ['cups_body_pre.xml','cups_body_day1.xml','cups_body_day2.xml','cups_body_day3.xml', 'cups_body_day4.xml', 'cups_body_day5.xml', 'cups_body_post.xml']
Datadir = 'Datafiles/'
df_all = []

for i in range(len(paramfiles)):
    
    Day_paramfile = Datadir + paramfiles[i]

    #Generate_PEACK_Data(Day_paramfile)
    res, res_labels = Analyze_PEACK_Data(Day_paramfile)
    df = ResultsDataFrame(res, res_labels, paramfiles[i], " ")
    #df_all.append(df)
    #df.to_csv(Datadir + "CUPS_day1.csv", index=False)
    #import pdb; pdb.set_trace()

#Generate_PEACK_Data(Datadir + "cups_pre.xml")
#Generate_PEACK_Data(Datadir + "cups_post.xml")

# res_pre, res_pre_labels = Analyze_PEACK_Data(Datadir + "cups_pre.xml")
# res_post, res_post_labels = Analyze_PEACK_Data(Datadir + "cups_post.xml")



# df1 = ResultsDataFrame(res_pre, res_pre_labels, "Pre", "2023")
# df2 = ResultsDataFrame(res_post, res_post_labels, "Post", "2023")
#import pdb; pdb.set_trace()
#df = pd.concat(df_all, ignore_index=True)
#df.to_csv(Datadir + "CUPS_analysis_out_v2.csv", index=False)


