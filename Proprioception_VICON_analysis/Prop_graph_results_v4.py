import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, kruskal, mannwhitneyu, median_test, normaltest, shapiro, brunnermunzel


def iqr(data):

    data = np.array(data)
    q75, q25 = np.percentile(data, [75,25])
    return q75 - q25

def plot_task(df, fname):
    sns.set_theme(style="darkgrid", font_scale=1)
    fig = plt.figure(figsize=(8.0, 6.0))
    ax = sns.boxplot(data=df, x="Group", y="Values", hue="Metric",
               linewidth=1, palette="colorblind", showfliers = False)
    sns.stripplot(data=df, x="Group", y="Values", hue="Metric",
                    palette='colorblind', linewidth=1, size=5, edgecolor='black', dodge=True)
    sns.despine(left=True)
    handles, labels = ax.get_legend_handles_labels()
    #plt.tight_layout()
    #plt.ylim(0,1)
    #plt.legend([],[], frameon=False)
    plt.legend(handles[0:3], labels[0:3], loc='lower right')
    plt.subplots_adjust(bottom=0.15)
    #plt.show()
    #plt.savefig(fname, pad_inches = 0, dpi=600)

def plot_task_dual(df, fname):

    
    sns.set_theme(style="darkgrid", font_scale=2)
    #fig = plt.figure(figsize=(8.0, 6.0))
    g = sns.catplot(data=df, x="Group", y="Values", hue="Metric", row='Pose',
               linewidth=1, palette="colorblind", showfliers = False, kind='box', height=4, aspect=9/4, legend=False)
    
    g.map_dataframe(sns.stripplot,x="Group", y="Values", hue="Metric",
                    palette='colorblind', linewidth=1, size=5, edgecolor='black', dodge=True)
    
    axes = g.axes.flatten()
    axes[0].set_title("Muscles Pose", pad=20)

    ax = plt.gca()
    ax.set_title("Powerbars Pose", pad=20)
    handles, labels = ax.get_legend_handles_labels()
    #plt.tight_layout()
    #plt.ylim(0,1)
    #plt.legend([],[], frameon=False)
    plt.legend(handles[0:3], labels[0:3], loc='lower right', fontsize=12)
    #plt.show()
    plt.savefig(fname, pad_inches = 0, dpi=600)


def plot_overall_symm(df2, fname):
    
    #import pdb; pdb.set_trace()
    sns.set_theme(style="darkgrid", font_scale=1)
    #fig = plt.figure(figsize=(8.0, 6.0))
    ax = sns.boxplot(data=df2, x="Group", y="Symmetry", palette = "Greys", hue = "Pose", linewidth=1, showfliers = False)
    sns.stripplot(data=df2, x="Group", y="Symmetry", palette = "Greys", hue = "Pose", dodge=True, linewidth=1, edgecolor='black')
    handles, labels = ax.get_legend_handles_labels()
    plt.ylim(0,1)
    plt.legend(handles[0:2], labels[0:2], loc='lower right')
    #plt.show()
    plt.savefig(fname, pad_inches = 0, dpi=600)

def get_combined_symmetry(df):
     
    Dist = df[df['Metric']=='Distance']['Values'].values
    Angle = df[df['Metric']=='Angle']['Values'].values
    Orient = df[df['Metric']=='Orientation']['Values'].values
    
    Total = Dist * Angle * Orient
    Group = df[df['Metric']=='Distance']['Group'].values
    Pose = df[df['Metric']=='Distance']['Pose'].values
    data = np.vstack((Group, Pose, Total)).T
    df2 = pd.DataFrame(data, columns=['Group', 'Pose', 'Symmetry'])

    df2['Symmetry'] = df2['Symmetry'].astype(float)
    return df2
    

def stat_compare_combined(data, pose, groups, stat):

    df_groups = []
    df_pose = data.loc[data['Pose'] == pose]
    
    for i in range(len(groups)):
        df_temp_grp = df_pose[df_pose['Group']==groups[i]]
        Dist = df_temp_grp[df_temp_grp['Metric']=='Distance']['Values'].values
        Angle = df_temp_grp[df_temp_grp['Metric']=='Angle']['Values'].values
        Orient = df_temp_grp[df_temp_grp['Metric']=='Orientation']['Values'].values
        # print('Dist normality p:', shapiro(Dist)[1])
        # print('Angle normality p:', shapiro(Angle)[1])
        # print('Orientation normality p:',shapiro(Orient)[1])
        val = Dist * Angle * Orient
        df_groups.append(val)

    #import pdb; pdb.set_trace()
    st = stat(*df_groups)
    stat_name = type(st).__name__
    stat_name = stat_name.replace('Result', '')
    print("Comparing groups: ", end=" ")
    print(*groups, sep=', ')
    print("Task: ", pose)
    print("Using statistic: ", stat_name)
    print("Statistic: ", st[0], " Sig: ", st[1])
    #import pdb; pdb.set_trace()
    print()


def stat_compare(data, pose, metric, groups, stat, stat_args='two-sided', stat2=np.median, stat3 = iqr):

    df_groups = []
    df_pose = data.loc[data['Pose'] == pose]
    df_metric = df_pose.loc[df_pose['Metric']==metric]
    for i in range(len(groups)):
        df_temp_grp = df_metric[df_metric['Group']==groups[i]]
        df_groups.append(df_temp_grp['Values'])

    #import pdb; pdb.set_trace()
    st = stat(*df_groups, alternative=stat_args)
    stat_name = type(st).__name__
    stat_name = stat_name.replace('Result', '')

    _, p1 = shapiro(df_groups[0])
    _, p2 = shapiro(df_groups[1])
    print("Comparing groups: ", end=" ")
    print(*groups, sep=', ')
    print("Task: ", pose)
    print("Measure compared: ", metric)
    print("Normalcy test for group1:", p1)
    print("Normalcy test for group2:", p2)
    print("Mean/Median of group 1: ", stat2(df_groups[0]), " +/- ", stat3(df_groups[0]))
    print("Mean/Median of group 2: ", stat2(df_groups[1]), " +/- ", stat3(df_groups[1])) 
    print("Using statistic: ", stat_name)
    print('Type of comparison:', stat_args)
    print("Statistic: ", st[0], " Sig: ", st[1])
    #import pdb; pdb.set_trace()
    print()

def stats_all_poses_combined(group_pair, selected_stat, stat2):
    
    stat_compare_combined(df,'Muscles', group_pair, selected_stat, stat2)
    stat_compare_combined(df,'PowerBars', group_pair, selected_stat, stat2)

def stats_all_poses_metrics(group_pair, selected_stat, selected_stat_args, stat2, stat3):

        stat_compare(df,'Muscles', 'Distance', group_pair, selected_stat, selected_stat_args, stat2, stat3)
        stat_compare(df,'Muscles', 'Angle', group_pair, selected_stat, selected_stat_args, stat2, stat3)
        stat_compare(df,'Muscles', 'Orientation', group_pair, selected_stat, selected_stat_args, stat2, stat3)
        stat_compare(df,'PowerBars', 'Distance', group_pair, selected_stat, selected_stat_args, stat2, stat3)
        stat_compare(df,'PowerBars', 'Angle', group_pair, selected_stat, selected_stat_args, stat2, stat3)
        stat_compare(df,'PowerBars', 'Orientation', group_pair, selected_stat, selected_stat_args, stat2, stat3)


def compare_two_groups(df, Groups, FigDir, selected_stat, selected_stat_args, stat2, stat3):
    
    fig_fname = FigDir + Groups[0] + '_vs_' + Groups[1]
    df_1 = df.loc[df['Group'] == Groups[0]]
    df_2 = df.loc[df['Group'] == Groups[1]]
    frames = [df_1, df_2]
    df = pd.concat(frames)

    df_muscles = df.loc[df['Pose'] == "Muscles"]
    df_powerbars = df.loc[df['Pose'] == "PowerBars"]
    #import pdb; pdb.set_trace()
    #plot_task(df_muscles, fig_fname + '_mus.' + ext)
    #plot_task(df_powerbars, fig_fname + '_pb.' + ext)
    plot_task_dual(df, fig_fname + '_dual.' + ext)
    #stats_all_poses_combined(Groups, selected_stat)
    stats_all_poses_metrics(Groups, selected_stat, selected_stat_args, stat2, stat3)
    

def compare_all_groups(df, FigDir):
    
    fig_fname = FigDir + 'All_groups'
    df_muscles = df.loc[df['Pose'] == "Muscles"]
    df_powerbars = df.loc[df['Pose'] == "PowerBars"]
    plot_task(df_muscles, fig_fname + '_mus.' + ext)
    plot_task(df_powerbars, fig_fname + '_pb.' + ext)
    
def compare_poses(df, Group, metrics, stat, stat2):

    df_poses = []
    df1 = df.loc[df['Group'] == Group]
    for i in range(len(metrics)):
        df1_met = df1.loc[df1['Metric']==metrics[i]]
        df_mus = df1_met.loc[df1_met['Pose']=='Muscles']
        df_pb = df1_met.loc[df1_met['Pose']=='PowerBars']
        x = df_mus['Values'].values
        y = df_pb['Values'].values
        
        st = stat(x,y)
        stat_name = type(st).__name__
        stat_name = stat_name.replace('Result', '')
        print("Comparing Muscles vs PowerBars in group", Group)
        print("Measure compared: ", metrics[i])
        print("Mean/Median of group 1: ", stat2(x), " +/- ", np.std(x))
        print("Mean/Median of group 2: ", stat2(y), " +/- ", np.std(y)) 
        print("Using statistic: ", stat_name)
        print("Statistic: ", st[0], " Sig: ", st[1])
        print()


#def heatmap_metrics(df, Group, FigDir):
     

FigDir = 'Figures/'
metaDataDir = 'Datafiles/'
#Fname = "Prop_Kids_Adults_mirror4.csv"
#Fname = "Prop_Kids_Adults_mirror6_20240301151812.csv"
Fname = "Prop_Kids_Adults_mirror7_20240910160057.csv"
ext = 'svg'
selected_stat = mannwhitneyu#ttest_ind#kruskal#brunnermunzel
sel_stat2 = np.median#np.mean
sel_stat3 = iqr
df = pd.read_csv(metaDataDir + Fname)
Metrics=['Angle', 'Distance', 'Orientation']
#import pdb; pdb.set_trace()

#df2 = get_combined_symmetry(df)

#plot_overall_symm(df2, FigDir + 'Combined_symmetry.' + ext)
#compare_all_groups(df, FigDir)

#compare_poses(df, 'CP LA', Metrics, selected_stat, sel_stat2)

#compare_two_groups(df, ['CP AH', 'CP LA'], FigDir, selected_stat, selected_stat_args='two-sided', stat2=sel_stat2, stat3=sel_stat3)
#compare_two_groups(df, ['CP AH', 'TD Control'], FigDir, selected_stat, selected_stat_args='less', stat2=sel_stat2, stat3=sel_stat3)
#compare_two_groups(df, ['CP LA', 'TD Control'], FigDir, selected_stat, selected_stat_args='less', stat2=sel_stat2, stat3=sel_stat3)
compare_two_groups(df, ['TD Control', 'Adult Control 2'], FigDir, selected_stat, selected_stat_args='two-sided', stat2=sel_stat2, stat3=sel_stat3)

#compare_two_groups(df, ['Adult Control 1', 'Adult Control 2'], FigDir, selected_stat, sel_stat2, sel_stat3)