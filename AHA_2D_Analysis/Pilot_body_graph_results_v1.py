import os
import scipy
import numpy as np
import sys
import pickle
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd
from scipy.stats import ttest_ind, wilcoxon

sys.path.insert(1, 'C:\\Users\\shiva\\Dropbox\\Burke Work\\DeepMarker\\Processed Data\\BBT_Data\\Scripts\\ReachingValidation\\')
sns.set()

def plotboxes(df1, df2, labels, fname):
    sns.set_theme(style="darkgrid", font_scale=2)
    f, (ax1,ax2) = plt.subplots(ncols=2, nrows=1, sharey=True)

    # Compare avg vel in dom vs non-dom hand
    g1 = sns.boxplot(data=df1,
                        whis=[0, 100], width=.8, palette="vlag", ax=ax1)


    g2 = sns.stripplot(data=df1, #hue='Subject',
                          size=7,  linewidth=0, ax=ax1)


    g3 = sns.boxplot(data=df2,
                    whis=[0, 100], width=.8, palette="vlag", ax=ax2)

    g4 = sns.stripplot(data=df2, #hue='Subject',
                      size=7, linewidth=0, ax=ax2)

    g1.set(ylabel=labels[0], xlabel="", title="Comfortable")
    g2.set(xticklabels=[labels[1], labels[2]])
    g3.set(xlabel="", title="Fast")
    g4.set(xticklabels=[labels[1], labels[2]])

    plt.subplots_adjust(wspace=0.05, hspace=0)
    plt.legend([],[], frameon=False)
    ax1.tick_params(axis='x', which='major', labelsize=20)
    ax2.tick_params(axis='x', which='major', labelsize=20)
    sns.despine(trim=True, left=True)
    plt.setp(ax1.get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.setp(ax2.get_xticklabels(), rotation=30, horizontalalignment='right')
    f.savefig("Figures//" + fname, dpi=600, bbox_inches='tight')
    #plt.show()

with open('KineMetrics_nf2.pkl', 'rb') as f:
    KineMetrics = pickle.load(f)

KineX = np.array(KineMetrics)

# Features: [mu_speed_dom, mu_speed_nondom, max_speed_dom, max_speed_nondom, mu_var_dom, mu_var_non_dom]

ComfData = KineX[:,0,:]
FastData = KineX[:,1,:]
df1 = pd.DataFrame(ComfData, columns = ['mu_speed_dom', 'mu_speed_nondom', 'max_speed_dom', 'max_speed_nondom', 'mu_var_dom', 'mu_var_non_dom'])
df2 = pd.DataFrame(FastData, columns = ['mu_speed_dom', 'mu_speed_nondom', 'max_speed_dom', 'max_speed_nondom', 'mu_var_dom', 'mu_var_non_dom'])

#Remove max speed info since it is still  sensitive to noise # To be fixed later
df1.drop(columns=['max_speed_dom', 'max_speed_nondom'], inplace=True)
df2.drop(columns=['max_speed_dom', 'max_speed_nondom'], inplace=True)
#Remove bad data record 5 for fast, 2 for comf
#df1.drop([2], inplace=True)
df2.drop([5], inplace=True)

print(df1)
print(df2)
#raise SystemExit
# Generate comparisons across avg speed
dfa = df1.drop(columns=['mu_var_dom', 'mu_var_non_dom'])
dfb = df2.drop(columns=['mu_var_dom', 'mu_var_non_dom'])
labels = ["Avg. Rate (Blks/sec)", "Dominant", "Non-Dominant"]
plotboxes(dfa, dfb, labels, "MeanSpeed.png")

#Generate trajectory variability
dfa = df1.drop(columns=['mu_speed_dom', 'mu_speed_nondom'])
dfb = df2.drop(columns=['mu_speed_dom', 'mu_speed_nondom'])
#dfa.drop([2], inplace=True)
labels = ["Avg. Range (cm)", "Dominant", "Non-Dominant"]
plotboxes(dfa, dfb, labels, "MeanTrajDev.png")

#Run Stats

def stat_test(X,Y):

    return ttest_ind(X,Y).pvalue

X = ComfData[:,0]
Y = ComfData[:,1]
print("Comf speed Dom vs Non-Dom: ", stat_test(X,Y))

X = FastData[:,0]
Y = FastData[:,1]
print("Fast speed Dom vs Non-Dom: ", stat_test(X,Y))

X = ComfData[:,0]
Y = FastData[:,0]
print("Comf vs Fast Dom Speed: ", stat_test(X,Y))


X = ComfData[:,1]
Y = FastData[:,1]
print("Comf vs Fast Non-Dom Speed: ", stat_test(X,Y))

X = ComfData[:,4]
Y = ComfData[:,5]
print("Comf Traj Dom vs Non-Dom: ", stat_test(X,Y))

X = FastData[:,4]
Y = FastData[:,5]
print("Fast Traj Dom vs Non-Dom: ", stat_test(X,Y))

X = ComfData[:,4]
Y = FastData[:,4]
print("Comf vs Fast Dom Traj: ", stat_test(X,Y))

X = ComfData[:,5]
Y = FastData[:,5]
print("Comf vs Fast Non-Dom Traj: ", ttest_ind(X,Y).pvalue)

#import pdb; pdb.set_trace()
