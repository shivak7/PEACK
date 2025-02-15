import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


def stat_plot(df, x_lab, y_lab, fn=[]):
    sns.lmplot(x=x_lab, y=y_lab, data=df)
    if len(fn) < 1:
        plt.show()
    else:
        plt.savefig(fn, pad_inches = 0, dpi=600)

    x = df[x_lab].values
    y = df[y_lab].values
    idx = ~np.isnan(y)
    x = x[idx]
    y = y[idx]
    #import pdb; pdb.set_trace()
    stat = stats.pearsonr(x, y)
    print('R^2 : ', stat[0]**2)
    print('p value : ', stat[1])
    

def plot_mus_bbt_reg(df):
    sns.set_theme(style="darkgrid", font_scale=1)
    sns.lmplot(x="BBT AH", y="PROP SYMM AH MUS", data=df)
    #sns.lmplot(x="BBT LA", y="PROP SYMM AH MUS", data=df)
    #sns.lmplot(x="BBT AH", y="PROP SYMM LA MUS", data=df)
    #sns.lmplot(x="BBT LA", y="PROP SYMM LA MUS", data=df)
    plt.show()

def plot_pb_bbt_reg(df):
    sns.set_theme(style="darkgrid", font_scale=1)
    sns.lmplot(x="BBT AH", y="PROP SYMM AH PB", data=df)
    # sns.lmplot(x="BBT LA", y="PROP SYMM AH PB", data=df)
    # sns.lmplot(x="BBT AH", y="PROP SYMM LA PB", data=df)
    # sns.lmplot(x="BBT LA", y="PROP SYMM LA PB", data=df)
    plt.show()


def plot_mus_copm_reg(df):
    sns.set_theme(style="darkgrid", font_scale=2)
    
    #stat_plot(df, 'COPM', 'PROP SYMM AH MUS')
    stat_plot(df, 'COPM', "PROP SYMM LA MUS", FigDir + 'COPM_LA_MUS' + ext)
    #stat_plot(df, 'COPM', "PROP SYMM AH PB")
    #stat_plot(df, 'COPM', "PROP SYMM LA PB")
    #sns.lmplot(x="COPM", y="PROP SYMM LA PB", data=df)
    


FigDir = 'Figures/'
metaDataDir = 'Datafiles/'
ext = '.svg'
Fname = 'CP_scores.csv'
df = pd.read_csv(metaDataDir + Fname)


#stat_plot(df, 'COPM', "PROP SYMM LA MUS")#, FigDir + 'COPM_LA_MUS' + ext)
#stat_plot(df, 'COPM', "PROP SYMM LA PB")#, FigDir + 'COPM_LA_MUS' + ext)
#stat_plot(df, 'PROP SYMM AH MUS', 'PROP SYMM AH PB') # **
#stat_plot(df, 'PROP SYMM AH PB', 'PROP SYMM LA PB') # ns
#stat_plot(df, 'PROP SYMM AH PB', 'AHA') # ns
#stat_plot(df, 'PROP SYMM LA MUS', 'AHA') # ns
#stat_plot(df, 'PROP SYMM AH MUS', 'MACS') # *
#stat_plot(df, "BBT AH", "PROP SYMM AH MUS")# ns
#stat_plot(df, "JEB LA", "PROP SYMM AH PB")# ns
stat_plot(df, "PROP SYMM AH MUS", "AHA")

#import pdb; pdb.set_trace()