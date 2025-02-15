import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, kruskal, mannwhitneyu, median_test


def plot_task(df):
    sns.set_theme(style="darkgrid", font_scale=2)
    plt.figure()
    sns.boxplot(data=df, x="Group", y="Values", hue="Metric",
               linewidth=1, palette="vlag", showfliers = False)
    sns.stripplot(data=df, x="Group", y="Values", hue="Metric",
                    palette={"Distance": "b", "Angle": "r"}, dodge=True)
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

    st = stat(*df_groups)
    stat_name = type(st).__name__
    stat_name = stat_name.replace('Result', '')
    print("Comparing groups: ", end=" ")
    print(*groups, sep=', ')
    print("Task: ", pose)
    print("Measure compared: ", metric)
    print("Using statistic: ", stat_name)
    print("Statistic: ", st.statistic, " Sig: ", st.pvalue)
    #import pdb; pdb.set_trace()
    print()


def stats_all_poses_metrics(group_pair):

        stat_compare(df,'Muscles', 'Distance', group_pair, selected_stat)
        stat_compare(df,'Muscles', 'Angle', group_pair, selected_stat)
        stat_compare(df,'PowerBars', 'Distance', group_pair, selected_stat)
        stat_compare(df,'PowerBars', 'Angle', group_pair, selected_stat)


df = pd.read_csv("Prop_Kids_Adults_2.csv")
df_muscles = df.loc[df['Pose'] == "Muscles"]
df_powerbars = df.loc[df['Pose'] == "PowerBars"]
#plot_task(df_muscles)
#plot_task(df_powerbars)
selected_stat = kruskal#mannwhitneyu
group_pair = ['CP AH', 'TD Control']
stats_all_poses_metrics(group_pair)

#group_pair = ['TD Control', 'Adult Control 2']
#stats_all_poses_metrics(group_pair)

#stat_compare(df_muscles, 'Distance', 'CP AH', 'Adult Control 1', kruskal)
#stat_compare(df_muscles, 'Distance', 'CP AH', 'Adult Control 2', kruskal)

#
# df_mus_dist_CP.describe()
# df_mus_dist_TD.describe()
# df_mus_dist_Adult.describe()

# df_dist = df.loc[df['Metric']=="Distance"]
# df_dist2 = df_dist.loc[df_dist['Group']!="CP LA"]
#import pdb; pdb.set_trace()
