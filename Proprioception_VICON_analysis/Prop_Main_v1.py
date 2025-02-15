import sys
import datetime
sys.path.insert(1, '../PEACK_API')
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



def Analyze_VICON_Data(ParamFile):

    Prop_Parameters = LoadParameters(ParamFile)
    FuncList = []
    FuncList.append(Proprioception.distance_symmetry_metric)
    FuncList.append(Proprioception.angle_symmetry_metric)
    FuncList.append(Proprioception.orientation_symmetry_metric)
    #FuncList.append(Proprioception.orientation_symmetry_metric_VICON)
    Res, IDs = AnalyzeAsSingleEpochs(Prop_Parameters, FuncList)
    return Res, IDs

def ResultsDataFrame(ResultValues, DataLabel, SubIDS=[]):

    Poses = ['Muscles', 'PowerBars']
    Metrics = ['Distance', 'Angle', 'Orientation']              #Add angle metric as well
    #import pdb; pdb.set_trace()
    #ID = 
    Value = []
    Pose = []
    Metric = []
    Group = []
    SubjectID = []
    for i in range(len(ResultValues)):
        for j in range(len(ResultValues[i])):
            for k in range(len(ResultValues[i][j])):
                Value.append(ResultValues[i][j][k])
                Metric.append(Metrics[k])
                Pose.append(Poses[i])
                Group.append(DataLabel)
                SubjectID.append(SubIDS[i][j])
    df_list = list(zip(SubjectID, Value, Metric, Pose, Group))
    df = pd.DataFrame(df_list,
                  columns=['ID', 'Values', 'Metric', 'Pose', 'Group'])

    #import pdb; pdb.set_trace()
    return df



#----------------- Data processing ---------------------------------
def batch_generate(DataFolder = 'Datafiles/'):
    Generate_VICON_Data(DataFolder + "Prop_CP_AH.xml")
    Generate_VICON_Data(DataFolder + "Prop_CP_LA.xml")
    Generate_VICON_Data(DataFolder + "Prop_Ctrl_Kids.xml")
    Generate_VICON_Data(DataFolder + "Prop_Ctrl_Adult_2019.xml")
    Generate_VICON_Data(DataFolder + "Prop_Ctrl_Adult_2022.xml")

#------------------ Analysis----------------------------------------
def batch_analyze(DataFolder = 'Datafiles/'):

    
    cp_ah_kids_res, cp_ah_ids = Analyze_VICON_Data(DataFolder + "Prop_CP_AH.xml")
    cp_la_kids_res, cp_la_ids = Analyze_VICON_Data(DataFolder + "Prop_CP_LA.xml")

    ctrl_kids_res, ctrl_kids_ids = Analyze_VICON_Data(DataFolder + "Prop_Ctrl_Kids.xml")
    #ctrl_adults_res_2019, ctrl_adults1_ids = Analyze_VICON_Data(DataFolder + "Prop_Ctrl_Adult_2019.xml")
    ctrl_adults_res_2022, ctrl_adults2_ids = Analyze_VICON_Data(DataFolder + "Prop_Ctrl_Adult_2022.xml")

    df1 = ResultsDataFrame(cp_ah_kids_res, 'CP AH', cp_ah_ids)
    df2 = ResultsDataFrame(cp_la_kids_res, 'CP LA', cp_la_ids)
    df3 = ResultsDataFrame(ctrl_kids_res, 'TD Control', ctrl_kids_ids)
    #import pdb; pdb.set_trace()
    #df4 = ResultsDataFrame(ctrl_adults_res_2019, 'Adult Control 1', ctrl_adults1_ids)
    df5 = ResultsDataFrame(ctrl_adults_res_2022, 'Adult Control 2', ctrl_adults2_ids)

    df = pd.concat([df1, df2, df3, df5], ignore_index=True)
    #df = pd.concat([df1, df3, df5], ignore_index=True)

    ts = datetime.datetime.now()
    ts_str = ts.strftime('%Y%m%d%H%M%S')
    df.to_csv(DataFolder + "Prop_Kids_Adults_mirror7_" + ts_str + ".csv", index=False)
    # #import pdb; pdb.set_trace()


batch_generate()
batch_analyze()
