import numpy as np
import pickle
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from proprioception_validation_stats_vicon import permute, get_pval, rpermute
import scipy
from scipy import stats
#from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.decomposition import PCA, FactorAnalysis

"""
    Runs off of data from Generate_ViconData_Analysis_v3.py
"""
def loadFile(fn):
    with open(fn, 'rb') as input:
        Data = pickle.load(input)
    return Data

def PlotData(df,GroupName, fmt):

        #params = {'axes.labelsize': 18,'axes.titlesize':20, 'font.size': 20, 'legend.fontsize': 20, 'xtick.labelsize': 28, 'ytick.labelsize': 40}
        #plt.rcParams.update(params)
        df_pose_angle = df.drop(df[df.Method=="Drift"].index)
        df_stability = df.drop(df[df.Method!="Drift"].index)

        sns.set_theme(style="darkgrid", font_scale=2)
        #sns.color_palette("tab10")
        # Initialize the figure with a logarithmic x axis
        #f, ax = plt.subplots(figsize=(7, 6))
        f, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, sharey=False, gridspec_kw={'width_ratios': [2, 1]})
        ax2.yaxis.tick_right()

        # Plot the orbital period with horizontal boxes
        g1 = sns.boxplot(data=df_pose_angle,x='Method',y='Measures',
                    whis=[0, 100], width=.8, palette="vlag", ax=ax1)


        g2 = sns.stripplot(data=df_pose_angle,x='Method',y='Measures', #hue='Subject',
                      size=7, color=".3", linewidth=0, ax=ax1)

        g3 = sns.boxplot(data=df_stability,x='Method',y='Measures',
                    whis=[0, 100], width=.8, palette="vlag", ax=ax2)

        g4 = sns.stripplot(data=df_stability,x='Method',y='Measures', #hue='Subject',
                      size=7, color=".3", linewidth=0, ax=ax2)


        if(GroupName[0:2]=='CP'):
            g1.set(ylim=(0.45, 1))
            g2.set(ylim=(0.45, 1))
            g3.set(ylim=(-3, 30.1))
            g4.set(ylim=(-3, 30.1))
            g4.set_yticks(np.arange(0, 31, 6.0))

        else:
            g1.set(ylim=(0.8, 1))
            g2.set(ylim=(0.8, 1))
            g3.set(ylim=(0.0, 8))
            g4.set(ylim=(0.0, 8))


        plt.subplots_adjust(wspace=0.05, hspace=0)
        plt.legend([],[], frameon=False)
        # Tweak the visual presentation
        ax1.xaxis.grid(True)
        ax1.set(ylabel="Symmetry")
        ax1.set(xlabel="")
        #ax1.yaxis.set_major_locator(plt.MaxNLocator(5))

        ax2.set(ylabel="Distance (m)")
        ax2.set(xlabel="")
        ax2.yaxis.set_label_position("right")
        #ax2.yaxis.set_major_locator(plt.MaxNLocator(5))
        #ax.set(xlabel=GroupName)
        sns.despine(trim=True, left=True)
        f.savefig("..//Figures//Groups//" + GroupName + fmt, dpi=600, bbox_inches='tight')
        #plt.show(block=False)

def PlotGroupData(Data, GroupNames, fmt):                        #Plot raw data for each group as bar and strip plots
    Methods = ["Distance", "Angle", "Drift"]
    for i in range(len(Data)):

        plot_data = Data[i].T
        L = plot_data.shape[0]
        measures = np.hstack(plot_data.T)
        method = []
        idx = np.array(np.arange(0,plot_data.shape[0],1))
        subject=np.tile(idx,3)
        for j in range(0,len(Methods)):
            arr = [Methods[j]]*L
            method = method + arr
        data = {'Measures':measures, 'Method':method, 'Subject':subject}
        df = pd.DataFrame(data=data)
        PlotData(df,GroupNames[i],fmt)
        #import pdb; pdb.set_trace()

def PlotCorrelation(x,y,fn=None):

    R = np.corrcoef(x,y)[0,1]
    sns.set_theme(style="darkgrid", font_scale=2)
    f, ax = plt.subplots(figsize=(7, 6))

    #sns.residplot(x=x, y=y, lowess=True, color="b")
    g = sns.regplot(x=x, y=y,ci=None);
    g.set(ylim=(np.max((np.min(y)-0.1,0)), np.min((np.max(y)+0.1,1))))
    g.set(xlim=(np.max((np.min(x)-0.1,0)), np.min((np.max(x)+0.1,1))))
    ax.set(ylabel="")
    ax.set(xlabel="")
    #ax.set(xlabel=fn)
    ax.set(title="R = " + "{:.2f}".format(R))
    #sns.despine(trim=True, left=True)
    if(fn!=None):
        f.savefig("..//Figures//" + fn, dpi=600, bbox_inches='tight')
    else:
        plt.show(block=False)

def PlotCorr(Data, Group1,Group2,method1,method2, fn=None):

    x = Data[Group1][method1]  #  Pose symmetry for CP AH Powerbars
    y = Data[Group2][method2]  #  Pose symmetry for CP AH Muscles
    #print(np.corrcoef(x,y)[0,1])
    #print(x,y)
    PlotCorrelation(x,y,fn)


def Stats(Data):

    x = Data[0][0]
    y = Data[1][0]
    s,p = scipy.stats.brunnermunzel(x,y)
    Dist = permute(x,y, nsims=10000)
    p = get_pval(Dist, s, obs_ymax=1)
    print('P value:', p)

def PlotHeatMapGrid(Mat, Labels1, Labels2=None, fn=None, size=(12,10)):

    if(Labels2==None):
        Labels2 = Labels1

    sns.set_theme(style="darkgrid", font_scale=2)
    f, ax = plt.subplots(figsize=size)

    g = sns.heatmap(Mat, annot=True, cmap='vlag')
    g.set_xticklabels(Labels1)
    g.set_yticklabels(Labels2, va='center')

    if(fn!=None):
        f.savefig("..//Figures//" + fn, dpi=600, bbox_inches='tight')
    else:
        plt.show(block=False)


def PlotJointCorrHeatMap(Data1, Data2, Labels, fn=None):

    JointData = np.vstack((Data1,Data2))
    Msize = JointData.shape[0]
    CorrMat = np.zeros((Msize, Msize))
    PMat = np.zeros((Msize, Msize))
    for i in range(0, Msize):
        for j in range(0, Msize):
            R = np.corrcoef(JointData[i],JointData[j])[0,1]
            Rstat = rpermute(JointData[i], JointData[j],nsims=10000)
            p = get_pval(Rstat, R, obs_ymax=100)
            #import pdb; pdb.set_trace()
            CorrMat[i,j] = R
            PMat[i,j] = p
            #CorrMat[i,j] = scipy.stats.spearmanr(JointData[i],JointData[j])[0]

    PlotHeatMapGrid(CorrMat, Labels, None, fn)
    PlotHeatMapGrid(PMat, Labels, None, fn[:-4] + '_pvals' + fn[-4:])
    #import pdb; pdb.set_trace()

    #plt.show(block=True)

def plot_symmetries(P_Err,V_Err):

    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots()
    ax.plot(P_Err, 'b')
    ax.plot(V_Err, 'r')
    #plt.xticks(np.arange(1, 18, step=2))
    ax.set(xlabel='Subject', ylabel='Symmetry')
           #title='Intel realsense (PEACK) vs VICON')
    ax.grid()
    plt.tight_layout()

    #fig.savefig("Centroid_error_unaligned.png", dpi=600, bbox_inches='tight')
    ##fig.savefig("Centroid_error_validation.png", dpi=600, bbox_inches = 'tight')
    #fig.savefig("Centroid_error_validation_normal2.png", dpi=600, bbox_inches='tight')
    plt.show(block=True)

def plot_drifts(Drifts, AH):

    Drifts2 = Drifts.copy()
    for i in range(len(Drifts2)):
        for j in range(len(Drifts2[0])):
            if(AH[j]==0):
                Drifts2[i][j] = np.flip(Drifts2[i][j])

    for i in range(len(Drifts2)):
        X = np.matrix(Drifts2[i])
        x = X[:,0]
        y = X[:,1]
        plot_symmetries(x,y)
        print(stats.ttest_ind(x, y, equal_var = False))
    import pdb; pdb.set_trace()



def ScreePlot(vals):


    PC_values = np.arange(len(vals))
    plt.plot(PC_values, vals, 'ro-', linewidth=2)
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Proportion of Variance Explained')
    plt.show()

def FactorLoadings(components, fn):

    feature_names = ['Distance', 'Angle', 'Drift']

    f, ax = plt.subplots(figsize=(7, 5))
    vmax = np.abs(components).max()
    ax.imshow(components, cmap="RdBu_r", vmax=vmax, vmin=-vmax)
    ax.set_yticks(np.arange(len(feature_names)))
    if ax.is_first_col():
        ax.set_yticklabels(feature_names)
    else:
        ax.set_yticklabels([])
    ax.set_title('')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Comp. 1", "Comp. 2"])
    plt.tight_layout()
    plt.show()

def plot_clusters(Data1, Data2, l1, l2,fn):

    x = np.hstack((Data1[0],Data2[0]))
    y = np.hstack((Data1[1],Data2[1]))
    z = np.hstack((Data1[2],Data2[2]))
    X = np.matrix((x,y,z))
    Y = stats.zscore(np.array(X), axis = 1)
    X = np.matrix(Y)
    pca = PCA(n_components=2)
    pca.fit(X.T)
    h = np.hstack((np.zeros(Data1.shape[1]), np.ones(Data2.shape[1])))
    #sz = 10*np.ones(Data1.shape[1] + Data2.shape[1])

    sns.set_theme(style="darkgrid", font_scale=2)
    f, ax = plt.subplots(figsize=(7, 5))

    g = sns.scatterplot(x=x, y=y, hue=h, s=300, legend=False, palette=[l1,l2])
    g.set(xlim=(0.5,1))
    g.set(ylim=(0.5,1))

    cx = np.array([np.mean(Data1[0]),np.mean(Data2[0])])
    cy = np.array([np.mean(Data1[1]),np.mean(Data2[1])])
    ch = np.array([0, 1])
    dist = np.sqrt((cx[0] - cx[1])**2 + (cx[0] - cx[1])**2)

    g = sns.scatterplot(x=cx, y=cy, hue=ch, s=500, legend=False, marker='X', palette=[l1,l2])
    ax1 = g.axes
    ax1.plot(cx, cy, color='black', linewidth=5.0)
    ax1.text(np.mean(cx)+0.01, np.mean(cy)-0.01, "d=" + str(round(dist,2)))
    ax1.set(ylabel="Angle Symmetry")
    ax1.set(xlabel="Pose Symmetry")
    #g.legend_.remove()


    transformer = FactorAnalysis(n_components=2, random_state=0, rotation='varimax')
    X_transformed = transformer.fit_transform(X.T)
    print(transformer.components_)
    fn2 = fn[0:-4] + '_FA_' + fn[-4:]
    PlotHeatMapGrid(transformer.components_.T, ['Component 1', 'Component 2'], Labels2=['Distance', 'Angle', 'Drift'], fn=fn2, size=(7, 6))
    #components = X_transformed.components_.T

    print('Eig: ', pca.explained_variance_ratio_)
    print('EV: ', pca.components_)

    if(fn!=None):
        f.savefig("..//Figures//" + fn, dpi=600, bbox_inches='tight')
    else:
        plt.show(block=False)



Data = loadFile('Out.dat')

#Group plots
GroupNames = ["CP_AH_Powerbars", "Adult_Powerbars", "CP_AH_Muscles", "Adult_Muscles", "CP_LA_Powerbars", "CP_LA_Muscles"]
fmt = '.png'
#fmt = '.svg'
PlotGroupData(Data,GroupNames,fmt)


#Cluster & Factor Ananlysis plots
plot_clusters(Data[0], Data[1], "sandybrown", "deepskyblue", 'Cluster_Powerbars_CP_AH_VS_Adults' + fmt)
plot_clusters(Data[1], Data[3], "navy", "deepskyblue", 'Cluster_Adults_PB_VS_Mus' + fmt)
plot_clusters(Data[2], Data[3], "sandybrown", "deepskyblue", 'Cluster_Muscles_CP_AH_VS_Adults' + fmt)
plot_clusters(Data[0], Data[2], "sandybrown", "brown", 'Cluster_CP_AH_PB_VS_MUS' + fmt)


#Correlations Plot
Labels = ['Distance', 'Angle', 'Drift', 'Distance', 'Angle', 'Drift']
PlotJointCorrHeatMap(Data[0], Data[2], Labels, 'CP_AHCorr' + fmt)
PlotJointCorrHeatMap(Data[1], Data[3], Labels, 'AdultCorr' + fmt)

input("Press enter to exit...")
