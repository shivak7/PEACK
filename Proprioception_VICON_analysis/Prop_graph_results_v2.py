import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, kruskal, mannwhitneyu, median_test


def plot_task(df):
    sns.set_theme(style="darkgrid", font_scale=2)
    plt.figure()
    sns.boxplot(data=df, x="Group", y="Values", hue="Metric",
               linewidth=1, palette="pastel", showfliers = False)
    sns.stripplot(data=df, x="Group", y="Values", hue="Metric",
                    palette={"Distance": "b", "Angle": "r", "Orientation":"g"}, dodge=True)
    sns.despine(left=True)

    plt.legend([],[], frameon=False)
    plt.show()


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


df = pd.read_csv("Prop_Kids_Adults_mirror.csv")
df_muscles = df.loc[df['Pose'] == "Muscles"]
df_powerbars = df.loc[df['Pose'] == "PowerBars"]
plot_task(df_muscles)
plot_task(df_powerbars)
selected_stat = kruskal#mannwhitneyu
group_pair = ['CP AH', 'TD Control']
stats_all_poses_metrics(group_pair, selected_stat)
#import pdb; pdb.set_trace()
# TODO:
# Replace with direction cosines?
# eg. [1,0,0] , [0 1 1]
#
