import sys
sys.path.insert(1, '../PEACK_API')
#from PEACK import ProcessAsMultiEpochs, AnalyzeEpochs
#from PEACK import ProcessAsSingleEpochs, AnalyzeAsSingleEpochs
import PEACK
import VICON
from ParamDef import LoadParameters
from PEACKMetrics import reaching
from PEACKMetrics import AHA
from PEACKMetrics import CUPS
from PEACKMetrics import Proprioception
from PEACKMetrics import metrics
import pickle
import numpy as np
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import CubicSpline
from scipy import signal
import matplotlib.pyplot as plt
from pyCompare import blandAltman
# P1 = AHAParamsDemo()
# SaveParameters(P1, "test.xml")



def VICON_joint_remapper_custom(ViconBody):

    bool_array = [True, False]
    for i in range(2):

        ViconBody.get_filtered_data_by_default = bool_array[i]
        try:
            RWrist = (ViconBody["RWrist1"] + ViconBody["RWrist2"])/2.0
            ViconBody['RWrist1', 'RWrist'] = RWrist
        except ValueError as ve:
            if(len(ViconBody["RWrist1"])>1):
                RWrist = ViconBody["RWrist1"]
                ViconBody['RWrist1', 'RWrist'] = RWrist
            else:
                RWrist = ViconBody["RWrist2"]
                ViconBody['RWrist2', 'RWrist'] = RWrist

        try:
            LWrist = (ViconBody["LWrist1"] + ViconBody["LWrist2"])/2.0
            ViconBody['LWrist1', 'LWrist'] = LWrist
        except ValueError as ve:
            if(len(ViconBody["LWrist1"])>1):
                LWrist = ViconBody["LWrist1"]
                ViconBody['LWrist1', 'LWrist'] = LWrist
            else:
                LWrist = ViconBody["LWrist2"]
                ViconBody['LWrist2', 'LWrist'] = LWrist

        try:
            MidHip = (ViconBody["RHip"] + ViconBody["LHip"])/2.0
            ViconBody['LHip', 'MidHip'] = MidHip
        except ValueError as ve:
            if(len(ViconBody["LHip"])>1):
                MidHip = ViconBody["LHip"]
                ViconBody['LHip', 'MidHip'] = MidHip
            else:
                MidHip = ViconBody["RHip"]
                ViconBody['RHip', 'MidHip'] = MidHip

    ViconBody.get_filtered_data_by_default = True
    del ViconBody['RWrist1']; del ViconBody['LWrist1']
    del ViconBody['RWrist2']; del ViconBody['LWrist2']
    del ViconBody['RHip']; del ViconBody['LHip']
    return ViconBody

def Generate_PEACK_Data(ParamFile):
    ##Process Body Kinematics
    Parameters = LoadParameters(ParamFile)
    PEACK.ProcessAsSingleEpochs(Parameters)


def Generate_VICON_Data(ParamFile):
    ##Process Body Kinematics
    Parameters = LoadParameters(ParamFile)
    VICON.ProcessAsSingleEpochs(Parameters, remapFunction=VICON_joint_remapper_custom)


def timeseries_correlation(sig1, sig2, sample_lag):
    
    val = 0
    if sample_lag > 0:
        adjusted_sig1 = sig1[:-sample_lag]
        adjusted_sig2 = sig2[sample_lag:]
    elif sample_lag < 0:
        sample_lag = np.abs(sample_lag)
        adjusted_sig1 = sig1[sample_lag:]
        adjusted_sig2 = sig2[:-sample_lag]
    else:
        adjusted_sig1 = sig1
        adjusted_sig2 = sig2

    size_diff = len(adjusted_sig1) - len(adjusted_sig2)
    if size_diff > 0:
        adjusted_sig1 = adjusted_sig1[:-size_diff]
    elif size_diff < 0:
        adjusted_sig2 = adjusted_sig2[:size_diff]

    val = np.abs(np.corrcoef(adjusted_sig1,adjusted_sig2)[0,1])
    return val, adjusted_sig1, adjusted_sig2


def stat_validation_analysis(Param_file1, Param_file2, stat_measure, *args):

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
    
    stat_data = []
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
        stat_trial = []
        for j in range(nplots):   #Iterate through trials

            Obj1 = TempData1[i][j]
            Obj2 = TempData2[i][j]
            stat1 = stat_measure(Obj1, *args)
            stat2 = stat_measure(Obj2, *args)
            stat_trial.append([stat1, stat2])
        
        stat_data.append(stat_trial)
    import pdb; pdb.set_trace(); 
    #stat_data[0].pop(-1)
    return np.array(stat_data)


def ts_validation_analysis(Param_file1, Param_file2, ts_measure, *args):
     
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

            #Obj1.get_filtered_data_by_default = False
            #Obj2.get_filtered_data_by_default = False

            sig1 = ts_measure(Obj1, *args)
            sig2 = ts_measure(Obj2, *args)

            #import pdb; pdb.set_trace()
            
            re_fs = np.lcm(int(Obj1.fs),int(Obj2.fs))
            
            Obj1.time = Obj1.time/1000                  #Special insert for the case when data timestamps are in ms for PEACK instead of seconds.
            
            if(len(Obj1.time)== len(sig1) + 1):         #If there is len(timestamp) = len(signal) + 1 : length mismatch of 1 sample.
                Obj1.time = Obj1.time[:-1]
            
            try:
                spl_sig1 = CubicSpline(Obj1.time, sig1)
                spl_sig2 = CubicSpline(Obj2.time, sig2)
            except:
                import pdb; pdb.set_trace()
                continue


            t1 = np.linspace(Obj1.time[0],Obj1.time[-1],num = len(sig1)*int(re_fs/Obj1.fs))
            t2 = np.linspace(Obj2.time[0],Obj2.time[-1],num = len(sig2)*int(re_fs/Obj2.fs))

            sig1_re = spl_sig1(t1)
            sig2_re = spl_sig2(t2)
            len_diff = np.abs(len(sig2_re) - len(sig1_re))
            
            #sig1_re, sig2_re = relative_zero_padding(sig1_re, sig2_re)

            cross_corr = np.correlate(sig1_re, sig2_re, 'full')
            sample_lag = len(sig2_re) - np.argmax(cross_corr)#len(sig1_re) - np.argmax(cross_corr)
            sig_lag = sample_lag/re_fs
            corr_trial_vec[0,j], sig1_re, sig2_re = timeseries_correlation(sig1_re, sig2_re, sample_lag)
            print('Correlation b/w PEACK and VICON:', corr_trial_vec[0,j])
            
            if i==1:#i==0 and j==0:
                
                #Obj1.get_filtered_data_by_default = False
                #Obj2.get_filtered_data_by_default = False
                #shoulder_posture_angle =  np.arctan((Obj1['RShoulder'][:,1] - Obj1['LShoulder'][:,1])/(Obj1['RShoulder'][:,0] - Obj1['LShoulder'][:,0]))#*180/np.pi
                #back_posture_angle = np.arctan((Obj1['Neck'][:,0] - Obj1['MidHip'][:,0])/(Obj1['Neck'][:,1] - Obj1['MidHip'][:,1]))#*180/np.pi

                #correction_factor = shoulder_posture_angle - np.mean(shoulder_posture_angle)
                #print('Participant\'s shoulders are at ', np.mean(shoulder_posture_angle), '+/-', np.std(shoulder_posture_angle), ' degrees to the horizontal wrt camera.')
                fig = plt.figure(figsize=[12,8])
                plt.rcParams.update({'font.size': 22})
                ax = fig.gca()
                stat_str = ' (r=' +  str(round(corr_trial_vec[0,j],2)) +')'
                ax.set_title(ts_measure.__name__ + stat_str)
                plt.plot(t1[:len(sig1_re)],sig1_re*180/np.pi, lw=2)
                plt.plot(t1[:len(sig1_re)],sig2_re*180/np.pi, lw=2)
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Degrees')
                plt.tight_layout()
                #plt.figure()
                #plt.plot(Obj1.time, sig1)#/(correction_factor + 1))
                #plt.plot(Obj2.time, sig2)
                
                
                plt.show()
                Neck_Reference = (Obj1['RShoulder'] + Obj1['LShoulder'])/2.0
                Obj1.get_filtered_data_by_default = True
                Obj2.get_filtered_data_by_default = True

                
                import pdb; pdb.set_trace()

                # fig, ax = plt.subplots(3, 1)

                # plt.subplot(3,1,2)
                # plt.plot(Obj1.time, sig1*(180/np.pi))
                # plt.plot(Obj2.time, sig2*(180/np.pi))
                # ax[1].set_title('Original Filtered Signal')

                # plt.subplot(3,1,3)
                # print(corr_trial_vec[0,j])
                # plt.plot(sig1_re*(180/np.pi))
                # plt.plot(sig2_re*(180/np.pi))
                # ax[2].set_title('Cubic Spline Interpolated Signal')
                

                # Obj1.get_filtered_data_by_default = False
                # Obj2.get_filtered_data_by_default = False
                # sig1_raw = ts_measure(Obj1, *args)
                # sig2_raw = ts_measure(Obj2, *args)
                # #plt.figure()
                # plt.subplot(3,1,1)
                # plt.plot(Obj1.time, sig1_raw*(180/np.pi))
                # plt.plot(Obj2.time, sig2_raw*(180/np.pi))
                # ax[0].set_title('Original Unfiltered Signal')
                # plt.show()
                #dist = dtw.distance(sig1, sig2)
                #path = dtw.warping_path(sig1, sig2)
                #dtwvis.plot_warping(sig1, sig2, path)
                #import pdb; pdb.set_trace()
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


def summarize_ts_correlations(corr_coeffs):

    for cc_i in range(len(corr_coeffs)):
        print('Participant ', cc_i+1, ' mean corr coeff:', np.mean(corr_coeffs[cc_i]))

    data_lengths = [len(y[0]) for y in corr_coeffs]
    corr_map = np.zeros((len(data_lengths), np.max(data_lengths)))

    for i in range(len(data_lengths)):
        corr_map[i,:data_lengths[i]] = corr_coeffs[i][0]

    plt.pcolor(corr_map)
    plt.colorbar()
    plt.clim([0.85, 1.0])
    plt.show()


def bland_altman_formatted(x,y, fn):

    fig = plt.figure(figsize=[14,8])
    plt.rcParams.update({'font.size': 22})
    ax = fig.gca()
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
    blandAltman(x, y, ax=ax, title=fn.__name__,pointColour='blue')
    plt.tight_layout()
    plt.show()


## Main code here

Datadir = 'Datafiles/'
PEACK_Params = Datadir + "TD_Body_Params.xml"
VICON_Params = Datadir + "TD_Body_VICON_Params.xml"
#Generate_PEACK_Data(PEACK_Params)
#Generate_VICON_Data(VICON_Params)

#corr_coeffs_1 = ts_validation_analysis(PEACK_Params, VICON_Params, reaching.metrics_raw)
#corr_coeffs_2 = ts_validation_analysis(PEACK_Params, VICON_Params, reaching.elbow_angle, 'L')
##corr_coeffs_2 = ts_validation_analysis(PEACK_Params, VICON_Params, reaching.elbow_angle, 'R')

#corr_coeff = ts_validation_analysis(PEACK_Params, VICON_Params, AHA.trunk_rotation_angle, 'raw','')
#corr_coeff = ts_validation_analysis(PEACK_Params, VICON_Params, AHA.trunk_displacement_angle, 'raw','')
corr_coeff = ts_validation_analysis(PEACK_Params, VICON_Params, reaching.elbow_angle, 'raw','R')
#print(corr_coeff[0])
#print(corr_coeff[1])

#summarize_ts_correlations(corr_coeff)
#print("Bimanual Task: ", np.mean(corr_coeff[0]))
#print("Unimanual Task: ", np.mean(corr_coeff[1]))

#fn = reaching.elbow_angle #AHA.trunk_rotation_angle
#res_stats = stat_validation_analysis(PEACK_Params, VICON_Params,  fn, 'statistic','L') #CUPS.hand_coordination, AHA.trunk_displacement_angle, AHA.trunk_rotation_angle
#res_stats = stat_validation_analysis(PEACK_Params, VICON_Params,  fn, 'statistic','') #CUPS.hand_coordination, AHA.trunk_displacement_angle, AHA.trunk_rotation_angle
#print(res_stats)


# pk_bi = res_stats[0][:,0]
# vic_bi = res_stats[0][:,1]
# pk_uni = res_stats[1][:,0]
# vic_uni = res_stats[1][:,1]
#bland_altman_formatted(pk_bi,vic_bi, fn)
#bland_altman_formatted(pk_uni,vic_uni,fn)
#import pdb; pdb.set_trace()

#blandAltman(res_stats[1][:,0], res_stats[1][:,1])
# tda_stats = stat_validation_analysis(PEACK_Params, VICON_Params, AHA.trunk_displacement_angle)
# plt.plot(tda_stats[0][:,0])
# plt.plot(tda_stats[0][:,1])
# plt.figure()
# plt.plot(tda_stats[1][:,0])
# plt.plot(tda_stats[1][:,1])
# plt.show()
# print(tda_stats)
#prop_angle_stats = stat_validation_analysis(PEACK_Params, VICON_Params, Proprioception.angle_symmetry_metric)
#print(prop_angle_stats)
#summarize_ts_correlations(corr_coeffs_1)
#summarize_ts_correlations(corr_coeffs_2)

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
