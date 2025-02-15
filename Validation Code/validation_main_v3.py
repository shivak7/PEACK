import sys
sys.path.insert(1, '../PEACK_API')
#from PEACK import ProcessAsMultiEpochs, AnalyzeEpochs
#from PEACK import ProcessAsSingleEpochs, AnalyzeAsSingleEpochs
import PEACK
import VICON
from ParamDef import LoadParameters
from PEACKMetrics import reaching
import pickle
import numpy as np
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import CubicSpline
import resampy
from scipy import signal
import matplotlib.pyplot as plt
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis

# P1 = AHAParamsDemo()
# SaveParameters(P1, "test.xml")


def Generate_PEACK_Data(ParamFile):
    ##Process Body Kinematics
    AHA_Parameters = LoadParameters(ParamFile)
    PEACK.ProcessAsSingleEpochs(AHA_Parameters)


def Generate_VICON_Data(ParamFile):
    ##Process Body Kinematics
    AHA_Parameters = LoadParameters(ParamFile)
    VICON.ProcessAsSingleEpochs(AHA_Parameters)


def Analyze_PEACK_Data(ParamFile):

    Parameters = LoadParameters(ParamFile)
    FuncList = []
    FuncList.append(reaching.metrics)
    Res, IDs = PEACK.AnalyzeAsSingleEpochs(Parameters, FuncList)
    #Res = np.squeeze(np.array(Res))
    return Res, IDs

def Analyze_VICON_Data(ParamFile):

    Parameters = LoadParameters(ParamFile)
    FuncList = []
    FuncList.append(reaching.metrics)
    Res, IDs = VICON.AnalyzeAsSingleEpochs(Parameters, FuncList)
    #Res = np.squeeze(np.array(Res))
    return Res, IDs

def get_subjectID(Obj):
    fn = os.path.split(Obj.filename)[1]
    fn = fn.replace('-','_')
    sID = fn.split('_')[0]
    return sID

def joint_validation_analysis(Param_file1, Param_file2):
     
    Param1 = LoadParameters(Param_file1)
    Param2 = LoadParameters(Param_file2)
    with open(Param1.OutFile, 'rb') as f:
        TempData1 = pickle.load(f)

    with open(Param2.OutFile, 'rb') as g:
        TempData2 = pickle.load(g)

    L1 = len(TempData1[0])
    L2 = len(TempData2[0])

    if(L1 != L2):
        print('Error: Uneven number of files for each group! A 1:1 match between files is required for comparisons. Validation is not possible.')
        raise(SystemExit)
    
    corr_data = []
    for i in range(len(TempData1)):          #Iterate through participants
        Participants = []
        PID = []
        n_files1 = len(TempData1[i])
        n_files2 = len(TempData2[i])

        if(n_files1 != n_files2):
            print('Error: Uneven number of files within group ' + i+1 + '! A 1:1 match between files is required for comparisons. Validation is not possible.')
            raise(SystemExit)
        sig_lag = 0
        
        nplots = len(TempData1[i])
        subsize_cols = 5
        subsize_rows = int(np.ceil(nplots/subsize_cols))
        #plt.figure()
        corr_trial_vec = np.zeros((1, nplots))
        for j in range(nplots):   #Iterate through trials

            Obj1 = TempData1[i][j]
            Obj2 = TempData2[i][j]
            sID1 = get_subjectID(Obj1)
            sID2 = get_subjectID(Obj2)
            sig1 = reaching.metrics_raw(Obj1)#reaching.elbow_angle(Obj1)#
            sig2 = reaching.metrics_raw(Obj2)#reaching.elbow_angle(Obj2)#
            re_fs = np.lcm(int(Obj1.fs),int(Obj2.fs))
            #sig1_re = signal.resample(sig1, len(sig1)*int(re_fs/Obj1.fs))#resampy.resample(sig1, Obj1.fs, re_fs, filter='sinc_window', window=signal.hann)
            #sig2_re = signal.resample(sig2, len(sig2)*int(re_fs/Obj2.fs))#resampy.resample(sig2, Obj2.fs, re_fs, filter='sinc_window', window=signal.hann)
            try:
                spl_sig1 = CubicSpline(Obj1.time, sig1)
                spl_sig2 = CubicSpline(Obj2.time, sig2)
            except:
                import pdb; pdb.set_trace()

            t1 = np.linspace(Obj1.time[0],Obj1.time[-1],num = len(sig1)*int(re_fs/Obj1.fs))
            t2 = np.linspace(Obj2.time[0],Obj2.time[-1],num = len(sig2)*int(re_fs/Obj2.fs))

            sig1_re = spl_sig1(t1)
            sig2_re = spl_sig2(t2)
            
            #sig3_re = resampy.resample(sig2, Obj2.fs, Obj1.fs)
            #sig2_re = signal.resample(sig2, len(sig1))
            #cross_corr = np.correlate(sig1_re,sig2_re, 'full')
            #sample_lag = len(sig1_re) - np.argmax(cross_corr)
            #sig_lag = sample_lag/Obj1.fs

            # if sample_lag > 0:
            #     corr_trial_vec[0,j] = np.abs(np.corrcoef(sig1_re[:-sample_lag],sig2_re[sample_lag:])[0,1])
            # elif sample_lag < 0:
            #     sample_lag = np.abs(sample_lag)
            #     corr_trial_vec[0,j] = np.abs(np.corrcoef(sig1_re[sample_lag:],sig2_re[:-sample_lag])[0,1])
            # else:
            #     corr_trial_vec[0,j] = np.abs(np.corrcoef(sig1_re,sig2_re)[0,1])

            # if(len(Obj1.time)== len(sig1) + 1):         #If there is len(timestamp) = len(signal) + 1 : length mismatch of 1 sample.
            #     Obj1.time = Obj1.time[:-1]
            
#            import pdb; pdb.set_trace()
            if(j==6) and (i==5):
                #dist = dtw.distance(sig1, sig2)
                #path = dtw.warping_path(sig1, sig2)
                #dtwvis.plot_warping(sig1, sig2, path)
                import pdb; pdb.set_trace()
        #     plt.subplot(subsize_rows, subsize_cols, j+1)
        #     plt.plot(Obj1.time,sig1,'k')
        #     plt.plot(Obj2.time - sig_lag,sig2,'r')

        # plt.show()
        corr_data.append(corr_trial_vec)
            #import pdb; pdb.set_trace()
    return corr_data

def ResultsDataFrame(ResultValues, SubjectLabels, GroupLabel, YearLabel=''):

    Value = []
    Subject = []
    Task = []
    Tasks = ['Dominant', 'Non-dominant']
    Measures = []#["Trunk Sway Angle", "Trunk Rotation Angle", "Trunk Sway Distance"]
    Group = []
    for i in range(len(ResultValues)):          # Tasks
        for j in range(len(ResultValues[i])):       #Subjects
                Value.append(ResultValues[i][j][:])
                Task.append(Tasks[i])
                Group.append(GroupLabel)
                Subject.append(SubjectLabels[i][j])# + '_' + YearLabel)
    npVal = np.array(Value)
    df_list = list(zip(Subject, npVal[:,0], npVal[:,1], npVal[:,2], npVal[:,3], Task, Group))
    df = pd.DataFrame(df_list, columns=["Subject ID", "Peak Velocity", "Time to Peak Velocity", "Peak acceleration", "Smoothness", "Handedness", "Task Speed"])
    #import pdb; pdb.set_trace()
    return df



def ValidationComaprison(PEACK_results = [], VICON_results = []):

    for i in range(len(PEACK_results)):

        P = np.array(PEACK_results[i])
        V = np.array(VICON_results[i])

        p_nsamples = P.shape[0]
        p_nmeasures = P.shape[1]
        v_nsamples = V.shape[0]
        v_nmeasures = V.shape[1]

        if (p_nsamples!=v_nsamples) or (p_nmeasures!=v_nmeasures):
            print('Mismatch between number of samples/trials (or number of measures)!')
            raise SystemError
        
        corr_mat = np.zeros((1, p_nmeasures))
        for j in range(p_nmeasures):


            p_mat = P[:, j]
            v_mat = V[:, j]
            #import pdb; pdb.set_trace()
            corr_mat[0,j] = np.corrcoef(p_mat, v_mat)[0,1]

        print(corr_mat)
        #import pdb; pdb.set_trace()



## Main code here

Datadir = 'Datafiles/'
PEACK_Params = Datadir + "PEACK_Adults_reaching.xml"
VICON_Params = Datadir + "VICON_Adults_reaching.xml"
Generate_PEACK_Data(PEACK_Params)
Generate_VICON_Data(VICON_Params)

corr_coeffs = joint_validation_analysis(PEACK_Params, VICON_Params)

for cc_i in range(len(corr_coeffs)):
    print('Participant ', cc_i+1, ' mean corr coeff:', np.mean(corr_coeffs[cc_i]))

data_lengths = [len(y[0]) for y in corr_coeffs]
corr_map = np.zeros((len(data_lengths), np.max(data_lengths)))

for i in range(len(data_lengths)):
    corr_map[i,:data_lengths[i]] = corr_coeffs[i][0]

plt.pcolor(corr_map)
plt.colorbar()
plt.show()

#import pdb; pdb.set_trace()

#Res1, IDs1 = Analyze_PEACK_Data(PEACK_Params)
#Res2, IDs2 = Analyze_VICON_Data(VICON_Params)
# df1 = ResultsDataFrame(Res, IDs, "Comf")

# Res, IDs = Analyze_PEACK_Data(Datadir + "BBT_Control_Adults_Fast.xml")
# df2 = ResultsDataFrame(Res, IDs, "Fast")

# df = pd.concat([df1, df2], ignore_index=True)
# df.to_csv(Datadir + "BBT_Adult_Healthy_2023.csv", index=False)

# print()
# ValidationComaprison(PEACK_results = Res1, VICON_results = Res2)
