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
    Generate_VICON_Data(DataFolder + "Prop_CP_AH_unblinded.xml")
    
#------------------ Analysis----------------------------------------
def batch_analyze(DataFolder = 'Datafiles/'):
    
    cp_ah_kids_res, cp_ah_ids = Analyze_VICON_Data(DataFolder + "Prop_CP_AH_unblinded.xml")
    
    df1 = ResultsDataFrame(cp_ah_kids_res, 'CP AH', cp_ah_ids)
    

    #ts = datetime.datetime.now()
    #ts_str = ts.strftime('%Y%m%d%H%M%S')
    df1.to_csv(DataFolder + "CP_Unblinded_mirror6" +  ".csv", index=False)
    # #import pdb; pdb.set_trace()


batch_generate()
batch_analyze()
