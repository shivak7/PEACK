import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns




def plot_comparison(df, x, y, ylab, ylim=[]):

    
    sns.boxplot(data=df,x=x, y=y, linewidth=1, showfliers = False)
    sns.stripplot(data=df, x=x, y=y, dodge=False, linewidth=1, edgecolor='black')
    if len(ylim) > 0:
        plt.ylim(ylim)
    plt.ylabel(ylab)
    

def plot_comf(df):
    
    df_comf = df[df['Task Speed']=='Comf']
    fig = plt.figure()
    fig.suptitle('Comfortably-paced task')
    plt.subplot(2,2,1)
    plot_comparison(df_comf, x='Handedness', y='Peak Velocity', ylab='Peak Velocity (m/s)', ylim=[0,6])
    plt.subplot(2,2,2)
    plot_comparison(df_comf, x='Handedness', y='Time to Peak Velocity', ylab='Time to Peak Velocity (ms)')
    plt.subplot(2,2,3)
    plot_comparison(df_comf, x='Handedness', y='Peak acceleration', ylab='Avg Acceleration (m/$s^2$)', ylim=[0,15])
    plt.subplot(2,2,4)
    plot_comparison(df_comf, x='Handedness', y='Smoothness', ylab='No. Sub-movements')
    plt.tight_layout()
    plt.savefig(Datadir + 'Comf.svg', dpi=300)

def plot_fast(df):
    
    df_fast = df[df['Task Speed']=='Fast']
    fig = plt.figure()
    fig.suptitle('Fast-paced task')
    plt.subplot(2,2,1)
    plot_comparison(df_fast, x='Handedness', y='Peak Velocity', ylab='Peak Velocity (m/s)', ylim=[])
    plt.subplot(2,2,2)
    plot_comparison(df_fast, x='Handedness', y='Time to Peak Velocity', ylab='Time to Peak Velocity (ms)')
    plt.subplot(2,2,3)
    plot_comparison(df_fast, x='Handedness', y='Peak acceleration', ylab='Avg Acceleration (m/$s^2$)', ylim=[0,15])
    plt.subplot(2,2,4)
    plot_comparison(df_fast, x='Handedness', y='Smoothness', ylab='No. Sub-movements')
    plt.tight_layout()
    plt.savefig(Datadir + 'Fast.svg', dpi=300)

def plot_dom(df, category):
    
    df_ref = df[df['Handedness']==category]
    fig = plt.figure()
    fig.suptitle('Using ' + category +  ' Hand')
    plt.subplot(2,2,1)
    plot_comparison(df_ref, x='Task Speed', y='Peak Velocity', ylab='Peak Velocity (m/s)', ylim=[0,6])
    plt.subplot(2,2,2)
    plot_comparison(df_ref, x='Task Speed', y='Time to Peak Velocity', ylab='Time to Peak Velocity (ms)')
    plt.subplot(2,2,3)
    plot_comparison(df_ref, x='Task Speed', y='Peak acceleration', ylab='Avg Acceleration (m/$s^2$)', ylim=[0,15])
    plt.subplot(2,2,4)
    plot_comparison(df_ref, x='Task Speed', y='Smoothness', ylab='No. Sub-movements')
    plt.tight_layout()
    plt.savefig(Datadir + category + '.svg', dpi=300)

Datadir = 'Datafiles/'
df = pd.read_csv(Datadir + "BBT_Adult_Healthy_2023.csv")

plot_comf(df)
plot_fast(df)

plot_dom(df, 'Dominant')
plot_dom(df, 'Non-dominant')


plt.show()
#import pdb; pdb.set_trace()
