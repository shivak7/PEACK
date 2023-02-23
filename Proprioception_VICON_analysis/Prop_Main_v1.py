import sys
sys.path.insert(1, '/Users/shiva/Dropbox/Burke Work/DeepMarker/Processed Data/PythonScripts/PEACK_API')
from VICON import ProcessAsSingleEpochs, AnalyzeAsSingleEpochs
from ParamDef import SaveParameters, LoadParameters
from PEACKMetrics import Proprioception
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def Generate_VICON_Data(ParamFile):
    ##Process Body Kinematics
    Prop_Parameters = LoadParameters(ParamFile)
    ProcessAsSingleEpochs(Prop_Parameters)


def VICON_joint_remapper(ViconBody):

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
        RElb = (ViconBody["RElbRadial"] + ViconBody["RElbUlnar"])/2.0
    except ValueError as ve:
        if(len(ViconBody["RElbRadial"])>1):
            RElb = ViconBody["RElbRadial"]
        else:
            RElb = ViconBody["RElbUlnar"]

    try:
        LElb = (ViconBody["LElbRadial"] + ViconBody["LElbUlnar"])/2.0
    except ValueError as ve:
        if(len(ViconBody["LElbRadial"])>1):
            LElb = ViconBody["LElbRadial"]
        else:
            LElb = ViconBody["LElbUlnar"]

    Body['LShoulder'] = ViconBody['LDeltoid']
    Body['RShoulder'] = ViconBody['RDeltoid']
    Body['LElbow'] = LElb
    Body['RElbow'] = RElb
    Body['LWrist'] = LWrist
    Body['RWrist'] = RWrist

    return Body


def Analyze_VICON_Data(ParamFile):

    Prop_Parameters = LoadParameters(ParamFile)
    FuncList = []
    FuncList.append(Proprioception.distance_symmetry_metric_VICON)
    FuncList.append(Proprioception.angle_symmetry_metric_VICON)
    FuncList.append(Proprioception.mirror_symmetry)
    #FuncList.append(Proprioception.orientation_symmetry_metric_VICON)
    Res = AnalyzeAsSingleEpochs(Prop_Parameters, FuncList)
    return Res

def ResultsDataFrame(ResultValues, DataLabel):

    Poses = ['Muscles', 'PowerBars']
    Metrics = ['Distance', 'Angle', 'Orientation']              #Add angle metric as well

    Value = []
    Pose = []
    Metric = []
    Group = []
    for i in range(len(ResultValues)):
        for j in range(len(ResultValues[i])):
            for k in range(len(ResultValues[i][j])):
                Value.append(ResultValues[i][j][k])
                Metric.append(Metrics[k])
                Pose.append(Poses[i])
                Group.append(DataLabel)

    df_list = list(zip(Value, Metric, Pose, Group))
    df = pd.DataFrame(df_list,
                  columns=['Values', 'Metric', 'Pose', 'Group'])

    #import pdb; pdb.set_trace()
    return df



#----------------- Data processing ---------------------------------
def batch_generate():
    Generate_VICON_Data("Prop_CP_AH.xml")
    Generate_VICON_Data("Prop_CP_LA.xml")
    Generate_VICON_Data("Prop_Ctrl_Kids.xml")
    Generate_VICON_Data("Prop_Ctrl_Adult_2019.xml")
    Generate_VICON_Data("Prop_Ctrl_Adult_2022.xml")

#------------------ Analysis----------------------------------------
def batch_analyze():

    ctrl_kids_res = Analyze_VICON_Data("Prop_Ctrl_Kids.xml")
    cp_ah_kids_res = Analyze_VICON_Data("Prop_CP_AH.xml")
    cp_la_kids_res = Analyze_VICON_Data("Prop_CP_LA.xml")

    ctrl_adults_res_2019 = Analyze_VICON_Data("Prop_Ctrl_Adult_2019.xml")
    ctrl_adults_res_2022 = Analyze_VICON_Data("Prop_Ctrl_Adult_2022.xml")

    df1 = ResultsDataFrame(cp_ah_kids_res, 'CP AH')
    df2 = ResultsDataFrame(cp_la_kids_res, 'CP LA')
    df3 = ResultsDataFrame(ctrl_kids_res, 'TD Control')
    #import pdb; pdb.set_trace()
    df4 = ResultsDataFrame(ctrl_adults_res_2019, 'Adult Control 1')
    df5 = ResultsDataFrame(ctrl_adults_res_2022, 'Adult Control 2')

    df = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)
    #df = pd.concat([df1, df3, df5], ignore_index=True)

    df.to_csv("Prop_Kids_Adults_mirror.csv", index=False)
    # #import pdb; pdb.set_trace()


#batch_generate()
batch_analyze()
