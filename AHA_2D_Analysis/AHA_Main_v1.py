import sys
sys.path.insert(1, 'C:\\Users\\shiva\\Dropbox\\Burke Work\\DeepMarker\\Processed Data\\PythonScripts\\PEACK_API')
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
    FuncList.append(AHA.trunk_sway_angle)
    FuncList.append(AHA.trunk_rotation_angle)
    FuncList.append(AHA.trunk_sway_distance)
    Res = AnalyzeEpochs(AHA_Parameters, FuncList)

    Res = np.squeeze(np.array(Res))
    df = pd.read_excel("AHA_Scores.xlsx")

    AHASum = df["sum"].values
    AHAOrient = df["orients"].values
    KineSum = np.sum(Res,axis=1)
    scaler = StandardScaler()
    KS = KineSum[:,0]
    #KineSum = np.squeeze(scaler.fit_transform(KineSum.reshape(-1,1)))
    #import pdb; pdb.set_trace()
    reg = LinearRegression().fit(KS.reshape(-1,1), AHASum)
    #reg = LinearRegression().fit(, AHASum)

    print("Coefficients:", reg.coef_, reg.intercept_)
    print("Linear regression score for AHAsum vs KineSum:", reg.score(KS.reshape(-1,1), AHASum))
    KScorr = np.corrcoef(AHASum,KS)[0][1]
    KOcorr = np.corrcoef(AHAOrient,KS)[0][1]
    print("Kinematics vs AHA sum score:", KScorr)
    print("Kinematics vs AHA orient sub-score:", KOcorr)
    plt.scatter(KS, AHASum, color="black")
    plt.show()
    #np.savetxt("Orientation.csv", Res, delimiter=",", fmt='%.2f')


## Main code here

#Generate_PEACK_Data("AHA_Body_Params_2018.xml")
Analyze_PEACK_Data("AHA_Body_Params_2015.xml")
