from proprioception_validation_v2 import process_VICON, plot_symmetries
from proprioception_validation_stats_vicon import get_pval, permute
import numpy as np
#from scipy.stats import mannwhitneyu
import scipy as scipy
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc
from docx import Document
import calendar;
import time;

Methods = ['Pose', 'Angle','Stability']
V_Fs = 100

class DataSet():
    def __init__(self, dir, lab):
        self.dir = dir
        self.label = lab
        Result = []

def RunStats(VSym1, VSym2):

    s,p = scipy.stats.brunnermunzel(VSym1, VSym2)
    Stats = permute(VSym1, VSym2, nsims=10000)
    p = get_pval(Stats, s, obs_ymax=100)
    #print("Perm test p value:", p)
    print([np.mean(VSym1), np.std(VSym1), np.mean(VSym2), np.std(VSym2), p])
    return [np.mean(VSym1), np.std(VSym1), np.mean(VSym2), np.std(VSym2), p]

def RunAnalysis(dir):
    Results = []
    for i in range(0,len(Methods)):

        res = process_VICON(dir, V_Fs, Methods[i])
        Results.append(res)
    return np.array(Results)

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

CP_AH_SuperheroPowerbars_dir = '..\\Data\\CP\\CP_AH_SuperheroPowerbars'
TD_ADULT_SuperheroPowerbars_dir = '..\\Data\\Healthy\\SuperheroPowerbars'
CP_AH_SuperheroMuscles_dir = '..\\Data\\CP\\CP_AH_SuperheroMuscles'
TD_ADULT_SuperheroMuscles_dir = '..\\Data\\Healthy\\SuperheroMuscles'
CP_LA_SuperheroPowerbars_dir = '..\\Data\\CP\\CP_LA_SuperheroPowerbars'
CP_LA_SuperheroMuscles_dir = '..\\Data\\CP\\CP_LA_SuperheroMuscles'

Dirs = [CP_AH_SuperheroPowerbars_dir,
        TD_ADULT_SuperheroPowerbars_dir,
        CP_AH_SuperheroMuscles_dir,
        TD_ADULT_SuperheroMuscles_dir,
        CP_LA_SuperheroPowerbars_dir,
        CP_LA_SuperheroMuscles_dir
        ]

FinalResult = []

# Main code for processing all the raw kinematic data and extracting proprioception and drift measures
for i in range(0,len(Dirs)):
    print(np.round(100*(i/len(Dirs))), '% complete...')
    res = RunAnalysis(Dirs[i])
    FinalResult.append(res)
print('100', '% complete...')

save_object(FinalResult, 'Out.dat')
input("Press Enter to exit...")
