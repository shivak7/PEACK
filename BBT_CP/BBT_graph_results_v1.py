import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel



def plot_comparison(df, x, y, ylab, ylim=[]):

    sns.boxplot(data=df,x=x, y=y, linewidth=1, showfliers = False)
    sns.stripplot(data=df, x=x, y=y, dodge=False, linewidth=1, edgecolor='black')
    if len(ylim) > 0:
        plt.ylim(ylim)
    plt.ylabel(ylab)
    
    pre_vals = df[df['Group']=='Pre'][y].values
    post_vals = df[df['Group']=='Post'][y].values

    tstat,p = ttest_rel(pre_vals, post_vals)

    #import pdb; pdb.set_trace()
    print('BBT Pre vs Post using: ', df['Handedness'].values[0])
    print('Comparing kinematic measure:', y)
    print('Mean pre-value:', np.mean(pre_vals))
    print('Mean post-value:', np.mean(post_vals))
    print('P value:', p)

    #import pdb; pdb.set_trace()

def plot_pre_post(df, category):
    
    df_ref = df[df['Handedness']==category]
    fig = plt.figure()
    fig.suptitle('Using ' + category +  ' Hand')
    plt.subplot(2,2,1)
    plot_comparison(df_ref, x='Group', y='Peak Velocity', ylab='Peak Velocity (m/s)')#, ylim=[0,6])
    plt.subplot(2,2,2)
    plot_comparison(df_ref, x='Group', y='Time to Peak Velocity', ylab='Time to Peak Velocity (ms)')
    plt.subplot(2,2,3)
    plot_comparison(df_ref, x='Group', y='Peak acceleration', ylab='Avg Acceleration (m/$s^2$)')#, ylim=[0,15])
    plt.subplot(2,2,4)
    plot_comparison(df_ref, x='Group', y='Smoothness', ylab='No. Sub-movements')
    plt.tight_layout()
    #plt.savefig(Datadir + category + '.svg', dpi=300)
    plt.show()

Datadir = 'Datafiles/'
df = pd.read_csv(Datadir + "BBT_CP_01222025.csv")

#plot_pre_post(df, 'Less-affected Hand')
plot_pre_post(df, 'Affected Hand')




plt.show()
#import pdb; pdb.set_trace()
