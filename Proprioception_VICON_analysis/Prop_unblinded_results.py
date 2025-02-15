import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, kruskal, mannwhitneyu, median_test, normaltest, shapiro

def iqr(data):

    data = np.array(data)
    q75, q25 = np.percentile(data, [75,25])
    return q75 - q25

def df_compare(Pose, df_blind, df_unblind, stat, stat2=np.median, stat3=iqr):

    df_blind_pose = df_blind.loc[df_blind['Pose']==Pose]
    df_unblind_pose = df_unblind.loc[df_unblind['Pose']==Pose]
    
    metric_labels = ['Distance', 'Angle', 'Orientation']
    for m in metric_labels:

        df_blind_met = df_blind_pose.loc[df_blind_pose['Metric']== m]
        df_unblind_met = df_unblind_pose.loc[df_unblind_pose['Metric']== m]
        df_intersect = pd.merge(df_blind_met, df_unblind_met, how='inner', on='ID')
        
        #import pdb; pdb.set_trace()

        x = df_intersect['Values_x'].values
        y = df_intersect['Values_y'].values
        st = stat(x,y)
        stat_name = type(st).__name__
        print("Measure compared: ", m)
        print("Mean/Median of group 1: ", stat2(x), " +/- ", stat3(x))
        print("Mean/Median of group 2: ", stat2(y), " +/- ", stat3(y))
        print("Using statistic: ", stat_name)
        print("Statistic: ", st[0], " Sig: ", st[1])
        print()
        #import pdb; pdb.set_trace()

data_folder = 'Datafiles/'
blinded_data_fn = 'Prop_Kids_Adults_mirror6_20240301151812.csv'
unblinded_data_fn = 'CP_Unblinded_mirror6.csv'

df_blind_data = pd.read_csv(data_folder + blinded_data_fn)
df_cp_ah = df_blind_data.loc[df_blind_data['Group'] == 'CP AH']

df_unblind_data = pd.read_csv(data_folder + unblinded_data_fn)

sel_stat2 = np.median#np.mean
sel_stat3 = iqr
stat = mannwhitneyu#ttest_ind
df_compare('Muscles', df_cp_ah, df_unblind_data, stat, sel_stat2, sel_stat3)
df_compare('PowerBars', df_cp_ah, df_unblind_data, stat, sel_stat2, sel_stat3)




