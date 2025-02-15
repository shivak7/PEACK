import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, kruskal, mannwhitneyu, median_test


def plot_task(df, fname):
    sns.set_theme(style="darkgrid", font_scale=2)
    #fig, ax = plt.subplots()
    fig = plt.figure(figsize=(8.0, 6.0))
    sns.boxplot(data=df, x="Group", y="Values", hue="Metric",
               linewidth=1, palette="pastel", showfliers = False)
    sns.stripplot(data=df, x="Group", y="Values", hue="Metric",
                    palette={"Distance": "b", "Angle": "r", "Orientation":"g"}, dodge=True)
    sns.despine(left=True)
    #plt.tight_layout()
    plt.legend([],[], frameon=False)
    plt.subplots_adjust(bottom=0.15)
    #plt.show()
    plt.savefig(fname, pad_inches = 0, dpi=300)


def stat_compare(data, pose, metric, groups, stat):

    df_groups = []
    df_pose = data.loc[data['Pose'] == pose]
    df_metric = df_pose.loc[df_pose['Metric']==metric]
    for i in range(len(groups)):
        df_temp_grp = df_metric[df_metric['Group']==groups[i]]
        df_groups.append(df_temp_grp['Values'])

    #import pdb; pdb.set_trace()
    st = stat(*df_groups)
    stat_name = type(st).__name__
    stat_name = stat_name.replace('Result', '')
    print("Comparing groups: ", end=" ")
    print(*groups, sep=', ')
    print("Task: ", pose)
    print("Measure compared: ", metric)
    print("Using statistic: ", stat_name)
    print("Statistic: ", st[0], " Sig: ", st[1])
    #import pdb; pdb.set_trace()
    print()


def stats_all_poses_metrics(group_pair, selected_stat):

        # stat_compare(df,'Muscles', 'Distance', group_pair, selected_stat)
        # stat_compare(df,'Muscles', 'Angle', group_pair, selected_stat)
        stat_compare(df,'Muscles', 'Orientation', group_pair, selected_stat)
        # stat_compare(df,'PowerBars', 'Distance', group_pair, selected_stat)
        # stat_compare(df,'PowerBars', 'Angle', group_pair, selected_stat)
        stat_compare(df,'PowerBars', 'Orientation', group_pair, selected_stat)


def compare_two_groups(df, Groups, FigDir, selected_stat):
    
    fig_fname = FigDir + Groups[0] + '_vs_' + Groups[1]
    df_1 = df.loc[df['Group'] == Groups[0]]
    df_2 = df.loc[df['Group'] == Groups[1]]
    frames = [df_1, df_2]
    df = pd.concat(frames)

    df_muscles = df.loc[df['Pose'] == "Muscles"]
    df_powerbars = df.loc[df['Pose'] == "PowerBars"]
    plot_task(df_muscles, fig_fname + '_mus.' + ext)
    plot_task(df_powerbars, fig_fname + '_pb.' + ext)

    stats_all_poses_metrics(Groups, selected_stat)
    

def compare_all_groups(df, FigDir):
    
    fig_fname = FigDir + 'All_groups'
    df_muscles = df.loc[df['Pose'] == "Muscles"]
    df_powerbars = df.loc[df['Pose'] == "PowerBars"]
    plot_task(df_muscles, fig_fname + '_mus.' + ext)
    plot_task(df_powerbars, fig_fname + '_pb.' + ext)
    

#def heatmap_metrics(df, Group, FigDir):
     


FigDir = 'Figures/'
metaDataDir = 'Datafiles/'
Fname = "Prop_Kids_Adults_mirror2.csv"
ext = 'png'
selected_stat = kruskal#mannwhitneyu
df = pd.read_csv(metaDataDir + Fname)

compare_all_groups(df, FigDir)

# compare_two_groups(df, ['CP AH', 'CP LA'], FigDir, selected_stat)
# compare_two_groups(df, ['CP AH', 'TD Control'], FigDir, selected_stat)

# compare_two_groups(df, ['CP LA', 'TD Control'], FigDir, selected_stat)

# compare_two_groups(df, ['TD Control', 'Adult Control 1'], FigDir, selected_stat)
# compare_two_groups(df, ['TD Control', 'Adult Control 2'], FigDir, selected_stat)

# compare_two_groups(df, ['Adult Control 1', 'Adult Control 2'], FigDir, selected_stat)



