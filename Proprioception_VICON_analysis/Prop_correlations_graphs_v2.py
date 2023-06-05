import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


# def stat_plot(df, x_lab, y_lab, fn=[]):
#     sns.lmplot(x=x_lab, y=y_lab, data=df)
#     if len(fn) < 1:
#         plt.show()
#     else:
#         plt.savefig(fn, pad_inches = 0, dpi=600)

#     x = df[x_lab].values
#     y = df[y_lab].values
#     idx = ~np.isnan(y)
#     x = x[idx]
#     y = y[idx]
#     #import pdb; pdb.set_trace()
#     stat = stats.pearsonr(x, y)
#     print('R^2 : ', stat[0]**2)
#     print('p value : ', stat[1])

# def stat_plot(x, y, fn=[]):
#     #sns.lmplot(x=x.values, y=y.values)
#     sns.regplot(x=x, y=y)

#     if len(fn) < 1:
#         plt.show()
#     else:
#         plt.savefig(fn, pad_inches = 0, dpi=600)

#     x_val = x
#     y_val = y
#     idx = ~np.isnan(x_val)
#     x_val = x_val[idx]
#     y_val = y_val[idx]
#     #import pdb; pdb.set_trace()
#     stat = stats.pearsonr(x_val, y_val)
#     print('R^2 : ', stat[0]**2)
#     print('p value : ', stat[1])

def inter_prop_metric_regression(df_prop, df_score, Group1, Group2):

    



FigDir = 'Figures/'
metaDataDir = 'Datafiles/'
ext = '.svg'
Fname_scores = 'CP_scores.csv'
Fname = "Prop_Kids_Adults_mirror2.csv"
df = pd.read_csv(metaDataDir + Fname)
df_scores = pd.read_csv(metaDataDir + Fname_scores)
df_scores = df_scores.sort_values('Sub ID')

df_cp_ah  = df.loc[df['Group'] == 'CP AH']
df_mus = df_cp_ah.loc[df_cp_ah['Pose'] == "Muscles"]
df_pb = df_cp_ah.loc[df_cp_ah['Pose'] == "PowerBars"]

#temp = df_mus.loc[df_mus['Metric']=='Orientation']#["Values"]#; x = x.values
temp = df_pb.loc[df_pb['Metric']=='Orientation']
temp = temp.sort_values('ID')

prop_ids = temp['ID'].values
score_ids = df_scores['Sub ID'].values
_, idx1, idx2 = np.intersect1d(prop_ids, score_ids, return_indices=True)

#x = df_scores['BBT AH'].values[idx2]
x = df_scores['MACS'].values[idx2]
y = temp['Values'].values[idx1]

#y = df_scores['BBT AH']


#y = df_mus.loc[df_mus['Metric']=='Orientation']["Values"]#; y = y.values
stat_plot(x, y)

#import pdb; pdb.set_trace()
