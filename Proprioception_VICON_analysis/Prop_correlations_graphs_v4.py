import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib import rcParams
from matplotlib import gridspec

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

def stat_plot(x, y, fn=[]):
#     #sns.lmplot(x=x.values, y=y.values)
#     sns.regplot(x=x, y=y)

#     if len(fn) < 1:
#         plt.show()
#     else:
#         plt.savefig(fn, pad_inches = 0, dpi=600)

    x_val = x
    y_val = y
    import pdb; pdb.set_trace()
    idx = ~np.isnan(x_val)
    x_val = x_val[idx]
    y_val = y_val[idx]
    
    stat = stats.pearsonr(x_val, y_val)
    print('R^2 : ', stat[0]**2)
    print('p value : ', stat[1])

def intra_prop_dataframes(df_prop, Group):

    df_grp  = df_prop.loc[df_prop['Group'] == Group]
    
    Poses = ['Muscles', 'PowerBars']
    Metrics = ['Distance', 'Angle', 'Orientation']
    df_intra = []
    for i in range(len(Poses)):    
        df_metrics = []
        df_pose = df_grp.loc[df_grp['Pose'] == Poses[i]]
        for j in range(len(Metrics)):
            temp = df_pose.loc[df_pose['Metric']==Metrics[j]]
            temp = temp.sort_values('ID')
            df_metrics.append(temp)
        df_intra.append(df_metrics)
    #import pdb; pdb.set_trace()
    return df_intra

def corr_heatmap(df_intra, fname):

    Poses = ['Muscles', 'PowerBars']
    Metrics = ['Distance', 'Angle', 'Orientation']
    xyticklabels = Metrics.copy()
    xyticklabels.extend(xyticklabels)
    L = len(Poses)*len(Metrics)

    Pmap = np.zeros((L,L))
    Pmap_p = np.ones((L,L))
    for i in range(L):
        for j in range(L):
                met1_idx = int(i%3)
                pose1_idx = int(i/3)
                met2_idx = int(j%3)
                pose2_idx = int(j/3)

                x = df_intra[pose1_idx][met1_idx]
                y = df_intra[pose2_idx][met2_idx]
                #print(pose1_idx, met1_idx)
                #print(pose2_idx, met2_idx)
                #import pdb; pdb.set_trace()

                s_id1 = x['ID'].values
                s_id2 = y['ID'].values
                #import pdb; pdb.set_trace()
                _, idx1, idx2 = np.intersect1d(s_id1, s_id2, return_indices=True)
                x_val = x['Values'].values[idx1]
                y_val = y['Values'].values[idx2]
                stat = stats.pearsonr(x_val, y_val)
                Pmap[i,j] = stat[0]
                Pmap_p[i,j] = stat[1]
    
    fig = plt.figure()
    Pmap_p[Pmap_p <= 0.05] = 0
    upper_tri = np.tril(np.ones(Pmap_p.shape)).astype(bool)
    ax = sns.heatmap(Pmap, mask = upper_tri, cmap='coolwarm', annot=True, xticklabels=xyticklabels, yticklabels=xyticklabels, vmin=0, vmax=1)   #Pmap_p.astype(bool) |
    #ax = sns.heatmap(Pmap, cmap='coolwarm', annot=True, xticklabels=xyticklabels, yticklabels=xyticklabels, vmin=0, vmax=1)
    
    ax.plot([0, L, L, 0], [0, 0, L, 0], clip_on=False, color='black', lw=2)
    fig.text(0.07, 0.7, 'Powerbars', ha='center', va='center', rotation='vertical', fontsize='xx-large')
    fig.text(0.07, 0.3, 'Muscles', ha='center', va='center', rotation='vertical', fontsize='xx-large')
    fig.text(0.59, 0.05, 'Powerbars', ha='center', va='center', rotation='horizontal', fontsize='xx-large')
    fig.text(0.28, 0.05, 'Muscles', ha='center', va='center', rotation='horizontal', fontsize='xx-large')
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)
    plt.xticks(fontsize=12, rotation=-30)
    plt.yticks(fontsize=12, rotation=0)
    ax.invert_yaxis()
    #ax.invert_xaxis()
    
    # fig2 = plt.figure()
    # ax2 = sns.heatmap(Pmap_p, mask = upper_tri, cmap='coolwarm', annot=True, xticklabels=xyticklabels, yticklabels=xyticklabels, vmin=0, vmax=1)   #Pmap_p.astype(bool) |
    # ax2.invert_yaxis()
    #plt.show()
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.5)
    plt.savefig(FigDir + fname + ext, pad_inches = 0,  bbox_inches = "tight", dpi=300)
    
    #import pdb; pdb.set_trace()
    

def intra_prop_corr_graph(df_intra, fname):

    k = 1
    #plt.subplot(3,2,k)
    #fig = plt.figure()
    fig = plt.figure(figsize=(8.0, 6.0))
    sns.set_theme(style="darkgrid", font_scale=1)
    met_cmp_indices = np.array([[0,1], [1, 2], [0, 2]])
    Poses = ['Muscles', 'PowerBars']
    for i in range(len(df_intra)):

        for j in range(len(met_cmp_indices)):
            idx1 = met_cmp_indices[j][0]
            idx2 = met_cmp_indices[j][1]
            #import pdb; pdb.set_trace()
            x = df_intra[i][idx1]['Values'].values
            y = df_intra[i][idx2]['Values'].values

            x_lab = df_intra[i][idx1]['Metric'].values[0]
            y_lab = df_intra[i][idx2]['Metric'].values[0]
            stat = stats.pearsonr(x, y)
            ax = plt.subplot(2,3,k)
            ax.set_ylabel(y_lab)
            #if(j==1):
            ax.set_title('r='+str(np.round(stat[0], 2))) #Poses[i])
            #if(i==1):
            ax.set_xlabel(x_lab)
            g = sns.regplot(x=x, y=y)
            g.set_xlim([0.4, 1])
            g.set_ylim([0.4, 1])
            k = k+1
    #plt.show()
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.5)
    plt.savefig(FigDir + fname + ext, pad_inches = 0,  bbox_inches = "tight", dpi=300)
    
def df_corr_stat(df1, df2):

    df1 = df1.sort_values('ID')
    df2 = df2.sort_values('ID')
    df1_ids = df1['ID'].values
    df2_ids = df2['ID'].values
    _, idx1, idx2 = np.intersect1d(df1_ids, df2_ids, return_indices=True)
    x = df1['Values'].values[idx1]
    y = df2['Values'].values[idx2]

    stat = stats.pearsonr(x, y)
    print('R : ', stat[0])
    print('R^2 : ', stat[0]**2)
    print('p value : ', stat[1])



def cp_score_vs_prop_metric(df_scores, score, df_prop):

    df_prop = df_prop.sort_values('ID')
    df_scores = df_scores.sort_values('Sub ID')
    df1_ids = df_prop['ID'].values
    df2_ids = df_scores['Sub ID'].values
    _, idx1, idx2 = np.intersect1d(df1_ids, df2_ids, return_indices=True)
    x = df_prop['Values'].values[idx1]
    y = df_scores[score].values[idx2]

    idx = ~np.isnan(y)
    x_val = x[idx]
    y_val = y[idx]

    return x_val,y_val


def cp_scores_vs_prop_all(df_scores, df_prop):


    Metrics = ['Distance', 'Angle', 'Orientation']
    #Scores = ['BBT AH', 'BBT LA', 'JEB AH', 'JEB LA', 'COPM', 'MACS', 'AHA']
    Scores = ['BBT AH', 'BBT LA', 'JEB AH', 'JEB LA', 'MACS', 'AHA']
    ScoreStat = [0,0,0,0,1,1]
    Groups = ['CP LA', 'CP AH']
    Poses = ['Muscles', 'PowerBars']

    ScoreVsProp = np.zeros((len(Poses), len(Metrics)*len(Groups), len(Scores)))
    for m in range(len(Poses)):
        df_pose = df_prop.loc[df_prop['Pose'] == Poses[m]]
        
        
        for k in range(len(Groups)):
            df_cp  = df_pose.loc[df_pose['Group'] == Groups[k]]
            for i in range(len(Metrics)):
                for j in range(len(Scores)):

                    temp = df_cp[df_cp['Metric']==Metrics[i]]
                    score = Scores[j]
                    x,y = cp_score_vs_prop_metric(df_scores, score, temp)
                    
                    stat = stats.pearsonr(x, y)
                    if(ScoreStat[j]==1):
                        stat = stats.spearmanr(x,y)
                    
                    #print('R : ', stat[0])
                    #print('R^2 : ', stat[0]**2)
                    #print('p value : ', stat[1])
                    #plt.subplot(len(Metrics), len(Scores), k)
                    #ax = sns.regplot(x=x, y=y)
                    
                    #ax.set_title(Metrics[i] + ' vs.' + Scores[j] + '\np=' + str(round(stat[1],4)) + '; R=' + str(round(stat[0],4)))
                    ScoreVsProp[m,len(Metrics)*k+i,j] = round(stat[0],4)
        
    return ScoreVsProp
    

def prepare_corr_df(corr_matrix, figname, df_name):
    sns.set_theme(style="darkgrid", font_scale=1.5)
    Metrics = ['Distance', 'Angle', 'Orientation']
    #Scores = ['BBT AH', 'BBT LA', 'JEB AH', 'JEB LA', 'COPM', 'MACS', 'AHA']
    Scores = ['BBT AH', 'BBT LA', 'JEB AH', 'JEB LA', 'MACS', 'AHA']
    

    mat = corr_matrix
    AH_Mat = mat[0:3]
    LA_Mat= mat[3:]
    reordered_mat = np.hstack((AH_Mat, LA_Mat)).flatten()
    Sign = np.sign(reordered_mat)


    groupHeader = ['LA']*len(Scores) + ['AH']*len(Scores)
    groupHeader = groupHeader*3
    MetricHeader = [Metrics[0]]*len(Scores)*2 + [Metrics[1]]*len(Scores)*2 + [Metrics[2]]*len(Scores)*2
    #MetricHeader = MetricHeader*2
    ScoreHeader = Scores*6
    columnNames = ['R Value', 'Sign', 'Hand Used', 'Symmetry Metric', 'Clinical Score']
    
    df = pd.DataFrame(list(zip(np.abs(reordered_mat), Sign, groupHeader, MetricHeader, ScoreHeader)), columns=columnNames)
    df.to_csv(df_name)
    print(df)

    #fig, axes = plt.subplots(1,2, sharey=False, figsize=(3, 2))
    fig = plt.figure(figsize=(9, 6)) 
    
    gs = gridspec.GridSpec(1, 2,
         wspace=0.0, hspace=0.0, 
    ) 

        
    col0 = df[df['Hand Used']=='LA']
    col1 = df[df['Hand Used']=='AH']

    #import pdb; pdb.set_trace()

    ax= plt.subplot(gs[0,0])
    sns.barplot(data=col0, x='Clinical Score', y='R Value', hue='Symmetry Metric', ax=ax, palette='Reds')
    ax.set_xlim(0, 1)
    text_start = -0.2
    text_step = 0.25
    curr_text_y = text_start
    #for i, v in enumerate(reordered_mat[0:21]):
    #ax.text(0, 0, str(v), ha='center')
    
    for s in range(len(Scores)):
        for m in range(len(Metrics)):
            #tval = np.abs(reordered_mat[s + m*len(Scores)])
            idx = m*len(Scores) + s 
            tval = col0['R Value'].values[idx] * col0['Sign'].values[idx]
            #ax.text(np.abs(tval), curr_text_y, str(tval), ha='right')
            if tval < 0:
                ax.text(np.abs(tval), curr_text_y, ' (-) ', ha='right')
            curr_text_y += text_step
        curr_text_y += text_step

    ax.invert_xaxis()
    ax= plt.subplot(gs[0,1])
    sns.barplot(data=col1, x='R Value', y='Clinical Score', hue='Symmetry Metric', ax=ax, palette='Reds')
    ax.set_xlim(0, 1)
    ax.set_ylabel(None)
    ax.set_yticks([])
    ax.get_legend().remove()
    sns.despine(fig,left=True, bottom=True)
    plt.axvline(x = 0, color = 'k', label = 'axvline - full height', linewidth = 2)

    val_start_index = 21
    curr_text_y = text_start
    for s in range(len(Scores)):
        for m in range(len(Metrics)):
            #tval = np.abs(reordered_mat[val_start_index + s + m*len(Scores)])
            idx = m*len(Scores) + s 
            tval = col1['R Value'].values[idx] * col1['Sign'].values[idx]
            #ax.text(np.abs(tval), curr_text_y, str(tval), ha='left')
            if tval < 0:
                ax.text(np.abs(tval), curr_text_y, ' (-) ', ha='left')
            curr_text_y += text_step
        curr_text_y += text_step

    #plt.savefig('Temp.png')
    #import pdb; pdb.set_trace()
    #axes[1].set_yticklabels([])
    #plt.subplots_adjust(wspace=0)
    #plt.savefig(figname)
    plt.show()
    #First Figure for Muscles pose:



rcParams.update({'figure.autolayout': True})
FigDir = 'Figures/'
metaDataDir = 'Datafiles/'
ext = '.svg'
Fname_scores = 'CP_scores.csv'
#Fname = "Prop_Kids_Adults_mirror5.csv"
Fname = "Prop_Kids_Adults_mirror6_20240301151812.csv"
df = pd.read_csv(metaDataDir + Fname)
df_scores = pd.read_csv(metaDataDir + Fname_scores)
df_scores = df_scores.sort_values('Sub ID')

Groups_names = ['CP AH', 'CP LA', 'TD Control', 'Adult Control 2']

# for grps in Groups_names:
#     IntraCorr_Group = grps
#     df_intra = intra_prop_dataframes(df, IntraCorr_Group)
#     intra_prop_corr_graph(df_intra, IntraCorr_Group + ' Reg')
#     corr_heatmap(df_intra, IntraCorr_Group + ' Corr_updated')


SvP = cp_scores_vs_prop_all(df_scores, df)
prepare_corr_df(SvP[0], FigDir + 'MusclesClinicalCorr.svg', metaDataDir + 'MusclesClincalCorr.csv')
prepare_corr_df(SvP[1], FigDir + 'PowerbarsClinicalCorr.svg', metaDataDir + 'PowerbarsClinicalCorr.csv')

# df_cp_ah  = df.loc[df['Group'] == 'CP AH']
# df_mus = df_cp_ah.loc[df_cp_ah['Pose'] == "Muscles"]
# df_pb = df_cp_ah.loc[df_cp_ah['Pose'] == "PowerBars"]

# x = df_pb.loc[df_pb['Metric']=='Orientation']
# y = df_mus.loc[df_mus['Metric']=='Orientation']
# df_corr_stat(x, y)
