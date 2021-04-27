import numpy as np
import os
import pandas as pd
from PreProcess import PreProcess
from PEACK import get_PEACK_UL_prop
from VICON import get_VICON_UL_prop
from scipy.signal import savgol_filter
from scipy.signal import medfilt
from scipy.signal import butter, lfilter, freqz, filtfilt
from common import getNormal, proprioception_align, proprioception_error_combo, proprioception_error_combo_unaligned, Asymmetry_Error2, Asymmetry_Error, Reflection_Error, proprioception_angles, joint_stability
import matplotlib.pyplot as plt
from pyCompare import blandAltman

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data.T, padlen=150)
    y = y.T
    return y

def plot_error_curves(P_Err,V_Err, fn=None):

    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots()
    ax.plot(np.linspace(1, 17, 17), P_Err, 'b')
    ax.plot(np.linspace(1, 17, 17), V_Err, 'r')
    plt.xticks(np.arange(1, 18, step=2))
    plt.ylim((0.5, 1))
    ax.set(xlabel='Subject', ylabel='Symmetry (%)')#,
           #title='Intel realsense (PEACK) vs VICON')
    ax.legend(['PEACK','VICON'])
    ax.grid()
    plt.tight_layout()

    if(fn!=None):
        fig.savefig(fn, dpi=600, bbox_inches='tight')
    #fig.savefig("Centroid_error_unaligned.png", dpi=600, bbox_inches='tight')
    ##fig.savefig("Centroid_error_validation.png", dpi=600, bbox_inches = 'tight')
    #fig.savefig("Centroid_error_validation_normal2.png", dpi=600, bbox_inches='tight')
    plt.show(block=True)

def plot_symmetries(P_Err,V_Err):

    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots()
    ax.plot(P_Err, 'b')
    ax.plot(V_Err, 'r')
    #plt.xticks(np.arange(1, 18, step=2))
    ax.set(xlabel='Subject', ylabel='Symmetry (m)')
           #title='Intel realsense (PEACK) vs VICON')
    ax.grid()
    plt.tight_layout()

    #fig.savefig("Centroid_error_unaligned.png", dpi=600, bbox_inches='tight')
    ##fig.savefig("Centroid_error_validation.png", dpi=600, bbox_inches = 'tight')
    #fig.savefig("Centroid_error_validation_normal2.png", dpi=600, bbox_inches='tight')
    plt.show(block=True)

def plot_error_vs_NormAngle(Err,NA):

    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots()
    ax.plot(np.linspace(1, 17, 17), Err, color='tab:blue')
    plt.xticks(np.arange(1, 18, step=2))
    ax.set(xlabel='Subject', title='Intel realsense (PEACK) vs VICON')
    ax.set_ylabel('Asymmetry Difference(m)', color='tab:blue')
    ax.tick_params(axis='y', labelcolor='tab:blue')
    ax2 = ax.twinx()
    ax2.plot(np.linspace(1, 17, 17), NA, color='tab:red')
    ax2.set_ylabel('Angle between planes (degrees)', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax.grid()

    plt.tight_layout()

    #fig.savefig("Centroid_error_Angles_mismatch.png", dpi=600, bbox_inches='tight')
    ##fig.savefig("Centroid_error_validation.png", dpi=600, bbox_inches = 'tight')
    ##fig.savefig("Hausdorff_error_validation_normal2.png", dpi=600, bbox_inches='tight')
    plt.show(block=True)

def plot_Angle_Asymmetry(PA,VA):

    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots()
    ax.plot(np.linspace(1, 17, 17), PA, 'b')
    ax.plot(np.linspace(1, 17, 17), VA, 'r')
    plt.xticks(np.arange(1, 18, step=2))
    ax.set(xlabel='Subject', ylabel='Angle Asymmetry (%)',
           title='Intel realsense (PEACK) vs VICON')
    ax.grid()
    plt.tight_layout()

    fig.savefig("Figures//Limb_angles.png", dpi=600, bbox_inches='tight')
    ##fig.savefig("Centroid_error_validation.png", dpi=600, bbox_inches = 'tight')
    #fig.savefig("Centroid_error_validation_normal2.png", dpi=600, bbox_inches='tight')
    plt.show(block=True)

def plot_error_timeseries(P_assym, V_assym, t1, t2,title='Intel realsense (PEACK) vs VICON'):

    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots()
    #import pdb; pdb.set_trace()
    if(P_assym!=[]):
        ax.plot(t1, P_assym, 'b')
    if(V_assym!=[]):
        ax.plot(t2, V_assym, 'r')

    plt.ylim((0, 100))
    ax.set(xlabel='Time (s)', ylabel='Symmetry (%)',
       title=title)
    ax.grid()
    plt.tight_layout()

    plt.show(block=True)


def plot_curve(y,x=[]):
    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots()
    if(x!=[]):
        ax.plot(x,y)
    else:
        ax.plot(y)
    plt.show(block=True)

def validate_VICON_PEACK(P_dir, P_Fs, V_dir, V_Fs,alignment='aligned'):
    P_fileinfo = os.listdir(P_dir)
    V_fileinfo = os.listdir(V_dir)

    if len(P_fileinfo)!=len(V_fileinfo):
        print('Error: Mismatch in number of records for PEACK and VICON system')

    P_Err = []
    V_Err = []
    NA = []
    PA_Err = []
    VA_Err = []
    RotMatrices = []
    CenterCoords = []
    for i in range(0,len(P_fileinfo)):

        P_fn = P_dir + os.sep + P_fileinfo[i]
        V_fn = V_dir + os.sep + V_fileinfo[i]

        temp = pd.read_csv(P_fn);
        temp = temp.values
        P_time = temp[:, 0];
        P_data = temp[:, 1:]

        temp = pd.read_csv(V_fn,skiprows=4)
        temp = temp.values
        V_time = temp[:, 0]/V_Fs
        V_data = temp[:, 2:]
        del(temp)

        P_data = PreProcess(P_data)
        V_data = PreProcess(V_data)

        P_UpperLimb_data = get_PEACK_UL_prop(P_data, P_Fs, -P_Fs)
        V_UpperLimb_data = get_VICON_UL_prop(V_data, V_Fs, -V_Fs)/1000.0

        N_joints = 8

        for j in range(0,N_joints):

            joint = np.squeeze(P_UpperLimb_data[j, :, :])
            joint = medfilt(joint, [99,1])
            P_UpperLimb_data[j, :, :] = savgol_filter(joint, 29, 3,axis=0)

        V_assym = []

        for t in range(0,V_UpperLimb_data.shape[1]):
            joint = np.squeeze(V_UpperLimb_data[:,t,:])
            #err = Asymmetry_Error2(joint,'VICON')
            err = proprioception_angles(joint)
            #V_X0, V_Joints1, V_normal = getNormal(joint, 'VICON')
            #err = Reflection_Error(V_Joints1, V_normal, V_X0, 'VICON')
            V_assym.append(err)

        V_assym = np.array(V_assym)

        P_assym = []
        for t in range(0,P_UpperLimb_data.shape[1]):
            joint = np.squeeze(P_UpperLimb_data[:,t,:])
            #err = Asymmetry_Error2(joint,'VICON')
            err = proprioception_angles(joint)
            #P_X0, P_Joints1, P_normal = getNormal(joint, 'PEACK')
            #err =  Reflection_Error(P_Joints1, P_normal, P_X0, 'PEACK')
            P_assym.append(err)

        P_assym = np.array(P_assym)

        t1 = P_time[P_Fs-1:]
        t2 = V_time[V_Fs-1:]

        pinched = P_assym[-P_Fs*3:-P_Fs*1]
        P_Err.append(np.mean(pinched))

        pinched = V_assym[-V_Fs*3:-V_Fs*1]
        V_Err.append(np.mean(pinched))
        #plot_error_timeseries(100*P_assym, 100*V_assym, t1, t2)

    P_Err = np.array(P_Err)
    V_Err = np.array(V_Err)
    return P_Err, V_Err, NA, 0, 0




def process_VICON(dir, Fs, Method='Pose'):

    fileinfo = os.listdir(dir)
    #print(fileinfo)
    Vsym = []
    NA = []
    A_Err = []
    CenterCoords = []
    for i in range(0,len(fileinfo)): #range(3,4):#

        if fileinfo[i].endswith(".csv") != True:
            continue
        fn = dir + os.sep + fileinfo[i]
        #print(fn)
        temp = pd.read_csv(fn,skiprows=4)
        temp = temp.values
        time = temp[:, 0]/Fs
        data = temp[:, 2:]
        del(temp)

        data = PreProcess(data)

        UpperLimb_data = get_VICON_UL_prop(data, Fs, -Fs)/1000.0
        N_joints = 8

        for j in range(0,N_joints):
            joint = np.squeeze(UpperLimb_data[j, :, :])
            #joint_old = joint.copy()
            joint = butter_lowpass_filter(joint, 6, Fs, order=5)
            UpperLimb_data[j, :, :] = joint
            #plot_symmetries(joint_old,joint)

        assym = []
        if(Method!='Stability'):
            for t in range(0,UpperLimb_data.shape[1]): #range(8*Fs,8*Fs+1)
                joint = np.squeeze(UpperLimb_data[:,t,:])

                err = 0
                if(Method=='Pose'):
                    err = Asymmetry_Error2(joint,'VICON')
                else:
                    err = proprioception_angles(joint)

                #import pdb; pdb.set_trace()
                #V_X0, V_Joints1, V_normal = getNormal(joint, 'VICON')
                #err = Reflection_Error(V_Joints1, V_normal, V_X0, 'VICON')
                assym.append(err)

            assym = np.array(assym)

            t1 = time[Fs-1:]

            #plot_error_timeseries([], 100*assym, [], t1,fileinfo[i])
            pinched = assym[-Fs*3:-Fs*1]
            Vsym.append(np.mean(pinched))
            #import pdb; pdb.set_trace()
        else:
            Jt = UpperLimb_data[:,-Fs*3:-Fs*1,:]
            dist = 100*joint_stability(Jt) # Converting to cm
            Vsym.append(dist)
            t1 = time[-Fs*2:-Fs*1]
            #plot_symmetries( UpperLimb_data[2,:,0], UpperLimb_data[5,:,0])
            #plot_symmetries(Jt[2,:,0],Jt[5,:,0])

    Vsym = np.array(Vsym)
    return Vsym
