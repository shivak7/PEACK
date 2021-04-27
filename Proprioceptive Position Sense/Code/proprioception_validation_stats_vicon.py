# Developed by Shivakeshavan Ratnadurai Giridharan @ BNI.
# (c) PEACK 2020
#This code performs bootstrap testing of VICON vs PEACK
import numpy as np
import os
import pandas as pd
from PreProcess import PreProcess
from PEACK import get_PEACK_UL_prop
from VICON import get_VICON_UL_prop
from scipy.signal import savgol_filter
from common import getNormal, proprioception_align, proprioception_error_combo, proprioception_error_combo_unaligned
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as scipy

def run_bootstrap(N, delta_error, V_dir, V_Fs, alignment='aligned'):

    V_fileinfo = os.listdir(V_dir)
    V_Err = []
    V_Err_noise = []
    NA = []
    V_UpperLimb_data_arr = []
    DistData = []
    Steps = 10

    for i in range(0,len(V_fileinfo)):

        V_fn = V_dir + os.sep + V_fileinfo[i]

        temp = pd.read_csv(V_fn,skiprows=4);
        temp = temp.values
        V_time = temp[:, 0]/V_Fs
        V_data = temp[:, 2:]
        del(temp)

        V_data = PreProcess(V_data)

        V_UpperLimb_data_arr.append(get_VICON_UL_prop(V_data, V_Fs, -V_Fs)/1000.0)

    for i in range(0,N):

        V_Err_noise = []
        V_Err = []
        NA = []

        for j in range (0, len(V_fileinfo)):


            V_UpperLimb_data = V_UpperLimb_data_arr[j]

            #arr = np.random.normal(loc=0., scale=[0.009, 0.004, 0.002], size=V_UpperLimb_data.shape)
            V_UpperLimb_data_noise = V_UpperLimb_data #+ delta_error + 0.01*np.random.randn(V_UpperLimb_data.shape[0], V_UpperLimb_data.shape[1], V_UpperLimb_data.shape[2])

            N_joints = 8

            # for k in range(0, N_joints):
            #     joint = np.squeeze(V_UpperLimb_data_noise[k, :, :])
            #     V_UpperLimb_data_noise[k, :, :] = savgol_filter(joint, 49, 3, axis=0)

            V_MeanPos = np.squeeze(np.mean(V_UpperLimb_data,axis=1))
            V_MeanPos_noise = np.squeeze(np.mean(V_UpperLimb_data_noise,axis=1))
            #import pdb; pdb.set_trace()
            arr1 = np.random.normal(loc=0., scale=[0.009, 0.004, 0.002], size=V_MeanPos_noise.shape) #White noise error
            arr2 = np.random.normal(loc=0., scale=[0.01, 0.01, 0.01], size=V_MeanPos_noise.shape) #Joint mismatch error
            V_MeanPos_noise = V_MeanPos_noise + arr1 + arr2#0.009*np.random.randn(V_MeanPos_noise.shape[0],V_MeanPos_noise.shape[1])

            if(alignment=='aligned'):
                V_MeanPos_noise, V_MeanPos, _, _ = proprioception_align(V_MeanPos_noise, V_MeanPos)
                v_err_noise, v_err, norm_angle = proprioception_error_combo(V_MeanPos_noise, V_MeanPos)
            else:
                v_err_noise, v_err, norm_angle = proprioception_error_combo_unaligned(V_MeanPos_noise, V_MeanPos)

            V_Err_noise.append(v_err_noise)
            V_Err.append(v_err)
            NA.append(norm_angle)

        if (i+1)%(N/Steps)==0:
            print( (100*(i+1)/N), " % completed...")

        V_Err_noise = np.array(V_Err_noise)
        V_Err = np.array(V_Err)
        NA = np.array(NA)
        DistData.append(V_Err_noise - V_Err)

    return DistData

def permute(x,y, nsims=1000):

    l1 = len(x)
    l2 = len(y)
    l3 = l1 + l2;
    idx = np.arange(0,l3,1)
    u = np.hstack((x,y))

    Stat = []
    for i in range(0,nsims):
        idx1 = np.random.choice(range(l3), l1, replace=False)
        mask = np.isin(idx,idx1, invert=True)
        idx2 = idx[mask]
        xp = u[idx1]
        yp = u[idx2]
        s,p = scipy.stats.brunnermunzel(xp, yp)
        Stat.append(s)

    return np.array(Stat)

def rpermute(x,y, nsims=1000):

    l1 = len(x)
    l2 = len(y)
    l3 = l1 + l2;

    Stat = []
    for i in range(0,nsims):
        idx1 = np.random.choice(range(l1), l1, replace=False)
        idx2 = np.random.choice(range(l2), l2, replace=False)
        xp = x[idx1]
        yp = y[idx2]
        s = np.corrcoef(xp,yp)[0,1]
        Stat.append(s)

    return np.array(Stat)

def get_pval(Dist, Obs, obs_ymax=1):

    D = Dist
    O = Obs
    #D =  np.nan_to_num(D)
    D[D == -np.inf] = 0
    D[D == np.inf] = 0

    Y = np.sort(D)
    #import pdb; pdb.set_trace()
    # sns.set_style("darkgrid")
    # try:
    #     pl = sns.displot(D,kde=False)
    # except:
    #     import pdb; pdb.set_trace()

    #pl = sns.displot(D, kind="kde")
    #fig = pl.get_figure()
    idx1 = int(0.025*D.shape[0])-1
    idx2 = int(0.975 * D.shape[0])+1
    #plt.plot([Y[idx1], Y[idx1]], [0, obs_ymax],'r')
    #plt.plot([Y[idx2], Y[idx2]], [0, obs_ymax],'r')
    #plt.plot([O, O], [0, obs_ymax],'g')
    p1 = np.sum(Y<=O)/D.shape[0]
    p2 = np.sum(Y>O)/D.shape[0]
    p = 2*np.min([p1,p2])

    #plt.title('P value: ' + str(p))
    #fig.savefig("Figures//Bootstrap_Centroid_aligned.png", dpi=600, bbox_inches='tight')
    #plt.show(block=True)
    return p
