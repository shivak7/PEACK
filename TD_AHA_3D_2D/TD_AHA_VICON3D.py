import sys
sys.path.insert(1, '../PEACK_API')
#from PEACK import ProcessAsMultiEpochs, ProcessAsSingleEpochs, AnalyzeEpochs, AnalyzeAsSingleEpochs
from VICON import ProcessAsSingleEpochs, AnalyzeAsSingleEpochs
from ParamDef import AHAParamsDemo, SaveParameters, LoadParameters
from PEACKMetrics import AHA
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# P1 = AHAParamsDemo()
# SaveParameters(P1, "test.xml")


def Custom_VICON_joint_remapper(ViconBody):

    try:
        RWrist = (ViconBody["RWrist1"] + ViconBody["RWrist2"])/2.0
    except ValueError as ve:
        if(len(ViconBody["RWrist1"])>1):
            RWrist = ViconBody["RWrist1"]
        else:
            RWrist = ViconBody["RWrist2"]

    try:
        LWrist = (ViconBody["LWrist1"] + ViconBody["LWrist2"])/2.0
    except ValueError as ve:
        if(len(ViconBody["LWrist1"])>1):
            LWrist = ViconBody["LWrist1"]
        else:
            LWrist = ViconBody["LWrist2"]

    try:
        MidHip = (ViconBody["LHip"] + ViconBody["RHip"])/2.0
    except:
        import pdb; pdb.set_trace()

    ViconBody['LWrist1', 'LWrist'] = LWrist
    ViconBody['RWrist1', 'RWrist'] = RWrist
    ViconBody['LHip', 'MidHip'] = MidHip
    

    del ViconBody['RWrist1']; del ViconBody['LWrist1']
    del ViconBody['RWrist2']; del ViconBody['LWrist2']
    del ViconBody['RHip']; del ViconBody['LHip']; 
    return ViconBody



def Generate_VICON_Data(ParamFile):
    ##Process Body Kinematics

    AHA_Parameters = LoadParameters(ParamFile)
    ProcessAsSingleEpochs(AHA_Parameters, Custom_VICON_joint_remapper)


def Analyze_VICON_Data(ParamFile):

    ## Analyze body movements (Orientation: Shoulder + Trunk Movements)

    AHA_Parameters = LoadParameters(ParamFile)

    FuncList = []
    FuncList.append(AHA.trunk_displacement_angle)
    FuncList.append(AHA.trunk_rotation_angle)
    FuncList.append(AHA.trunk_displacement_distance)
    FuncList.append(AHA.elbow_flexion_angle)                                        #Temporary! change back to append for single metric
    Res, Labels = AnalyzeAsSingleEpochs(AHA_Parameters, FuncList)
    #import pdb; pdb.set_trace()
    Res = np.squeeze(np.array(Res))
    return Res, Labels
    #np.savetxt("Orientation.csv", Res, delimiter=",", fmt='%.2f')

def ResultsDataFrame(ResultValues, SubjectLabels, GroupLabel, YearLabel):

    Value = []
    Subject = []
    Task = []
    Measures = []#["Trunk Sway Angle", "Trunk Rotation Angle", "Trunk Sway Distance"]
    Group = []
    for i in range(len(ResultValues)):          #Tasks
        for j in range(len(ResultValues[i])):       # Subjects
                Value.append(ResultValues[i][j][:])
                Task.append("Task " + str(i + 1))
                Group.append(GroupLabel)
                Subject.append(SubjectLabels[i][j] + '_' + YearLabel)
    npVal = np.array(Value)
    df_list = list(zip(Subject, npVal[:,0], npVal[:,1], npVal[:,2], npVal[:,3], npVal[:,4], Task, Group))
    #import pdb; pdb.set_trace()
    df = pd.DataFrame(df_list,
                  columns=["Subject ID", "Trunk Angular Displacement", "Trunk Rotation Angle", "Trunk linear Distance", "Elbow Flexion L", "Elbow Flexion R", "Task", "Group"])

    #import pdb; pdb.set_trace()
    return df

## Main code here
Datadir = 'Datafiles/'
#Generate_VICON_Data(Datadir + "TD_Body_VICON_Params.xml")
resTD, resTD_labels = Analyze_VICON_Data(Datadir + "TD_Body_VICON_Params.xml")

df1 = ResultsDataFrame(resTD, resTD_labels, "TD", "2023")
#print(df1)
df1.to_csv(Datadir + "AHA_TD_VICON.csv", index=False)

#import pdb; pdb.set_trace()
