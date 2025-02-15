import os
import scipy
import numpy as np
import sys
import pickle
sys.path.insert(1, 'C:\\Users\\shiva\\Dropbox\\Burke Work\\DeepMarker\\Processed Data\\BBT_Data\\Scripts\\ReachingValidation\\')
#sys.path.insert(1, '//Users//shiva//Dropbox//Burke Work//DeepMarker//Processed Data//BBT_Data//Scripts//ReachingValidation//')

from DataLoader import DataLoader
from ExtractKinematicData import ExtractKinematicData
import variableDeclaration as var
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import PEACK_Filters as PF
import seaborn as sns
from scipy.signal import find_peaks, find_peaks_cwt, blackmanharris
from scipy import signal
from PEACKMetrics import kine
from PEACKMetrics import segment
#from sklearn.feature_selection import f_regression, mutual_info_regression
import nolds #nolds.lyap_r(z)


def ProcessLoop():

    Metrics = np.zeros((1,5))
    for task in TaskTypes:
        for hand in HandType:
            for id in range(0,len(PEACKData[task][hand])):
                M = ProcessData(task, hand, id)
                Metrics = np.vstack((Metrics, M))
    return np.matrix(Metrics[1:])
    # with open('KineMetrics_nf2.pkl', 'wb') as f:
    #     pickle.dump(KineMetrics, f)

def ProcessData(task, hand, id, filtered=True):

    hand_idx = -1;
    if(hand=='left'):
        hand_idx = LeftHandWrist
    elif (hand=='right'):
        hand_idx = RightHandWrist

    Sig3D = None
    fs = PEACKData[task][hand][id].fs

    if filtered==True:
        Sig3D = np.apply_along_axis(PF.center_and_clip, 0, PEACKData[task][hand][id].data_filtered[hand_idx,WindowStart:-WindowEnd,:], *SignalClippingRange)
    else:
        Sig3D = np.apply_along_axis(PF.center_and_clip, 0, PEACKData[task][hand][id].data[hand_idx,WindowStart:-WindowEnd,:], *SignalClippingRange)

    #Sig3D = np.apply_along_axis(PF.median_filter_vector, 0, Sig3D, *[60, 0.1])
    #segment.movements(Sig3D, 60)

    v,a = kine.va_process(Sig3D, fs, axis = 0)
    speed = np.linalg.norm(v,2, axis=1)
    seg_sig = -speed
    #seg_sig = Sig3D[:,1]

    tau = kine.autocorr_delay(seg_sig)                                           # "Quasi-period" estimation using auto correlation
    start_indices, stop_indices = segment.movements(seg_sig, fs,peak_prom=None, axis=1, dist=tau/4)  #Use "quasi-period" to estimate distance param for findpeaks
    segmented_t = segment.trials(Sig3D, start_indices, stop_indices)
    segmented_v = segment.trials(v, start_indices, stop_indices)
    segmented_a = segment.trials(a, start_indices, stop_indices)

    # print(PEACKData[task][hand][id].filename)
    # print("Delay: ", tau)
    # print("No. of Attempts: ", len(start_indices))
    #import pdb; pdb.set_trace()

    # plt.plot(-speed)
    # plt.plot(np.abs(Sig3D[:,0] + Sig3D[:,1] + Sig3D[:,2]))
    #
    # plt.show(block=True)
    Metrics = kine.Eval_trajectories(segmented_t, segmented_v, segmented_a, fs)
    #import pdb; pdb.set_trace()
    return Metrics

sns.set()
with open('AHA_2015_Initial.pkl', 'rb') as f:
    TempData = pickle.load(f)

import pdb; pdb.set_trace()

PEACKData = {
    'fast': {
        'left': TempData[0][::2],
        'right': TempData[0][1::2]
    }
}

KineMetrics = []
WindowStart = 1 + 1*60;
WindowEnd = 1*60;
LeftHandWrist = 5
RightHandWrist = 2
TaskTypes = ['fast']
HandType = ['left', 'right']
SignalClippingRange = [-200,200]
id = 0;
Fs = 60;

Metrics = ProcessLoop()
print(Metrics)

# with open('CP_analysis.pkl', 'wb') as f:
#     pickle.dump(Metrics, f)

#ProcessData('comf', 'right', 1)
#ProcessData('fast', 'left', 2)
#ProcessData('fast', 'right', 2)
