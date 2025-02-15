import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, ttest_rel, mannwhitneyu
from matplotlib import pyplot as plt
import seaborn as sns


def load_data_with_treatment_col(data_fn, treatment_fn):
    
    df_data = pd.read_csv(Datadir + data_fn)
    df_treatment = pd.read_csv(Datadir + treatment_fn) 

    data_subject_ids = list(df_data['Subject ID'].values)
    data_treatment_subject_ids = list(df_treatment['ID'].values)
    data_treatment_vals = df_treatment['Intervention (1= CIMT 2 = HABIT)'].values
    data_len = len(data_subject_ids)
    treatment_col = np.zeros((data_len,1))
    id_len = 7  #First 7 chars encode the subject id
    #data_subject_ids = [x.replace(" ", "") for x in data_subject_ids]
    #data_treatment_subject_ids = [x.replace(" ", "") for x in data_treatment_subject_ids]
    data_subject_ids = [x[0:id_len] for x in data_subject_ids]
    data_treatment_subject_ids = [x[0:id_len] for x in data_treatment_subject_ids]

    for i in range(data_len):
        
        treatment_idx = data_treatment_subject_ids.index(data_subject_ids[i])
        treatment_col[i] = data_treatment_vals[treatment_idx]

    df_out = df_data.assign(Treatment=treatment_col)
    return df_out


def generate_task_list(task_numbers):

    task_list = []
    for i in range(len(task_numbers)):
        task_list.append('Task ' + str(task_numbers[i]))
    return task_list

def select_from_tasks(df_data, task_list):

    df_res = df_data.loc[df_data['Task'].isin(task_list)]
    return df_res

def combine_across_tasks(df_data):

    uIDs = list(df_data['Subject ID'].unique())
    all_data_headers = list(df_data.columns)
    dcols = list(df_data.select_dtypes([np.float64]).columns)[:-1]
    cols = ['Subject ID','Group', 'Treatment']
    cols[1:1] = dcols
    combined_data_raw = []
    for i in range(len(uIDs)):

        curr_uid = uIDs[i]
        df2 = df_data.loc[df_data['Subject ID']==curr_uid]
        vals = df2.mean(numeric_only=True).values[:-1]
        temp_list = [curr_uid, df2['Group'].unique()[0], df2['Treatment'].unique()[0]]
        temp_list[1:1] = vals
        combined_data_raw.append(temp_list)
    df_combined = pd.DataFrame(combined_data_raw, columns=cols)
    return df_combined


def mad_outlier_screen(data_array, ad_scale=3):             #Median abs deviation based screening

    ad = np.nanmean(np.absolute(data_array - np.nanmean(data_array))) #absolute
    ub = np.nanmedian(data_array) + ad_scale*ad
    lb = np.nanmedian(data_array) - ad_scale*ad
    
    #import pdb; pdb.set_trace()
    valid_data_idx = (data_array < ub) & (data_array > lb)  #screen for outliers
    return valid_data_idx


def plot_pre_post_jitter(y_pre, y_post, figname=[], fps=1):
    plt.rcParams.update({'font.size': 22})
    jitter_width = 0.05
    x_pre = 0 + (jitter_width * (2*np.random.rand(len(y_pre),) - 1))
    x_post = 1 + (jitter_width * (2*np.random.rand(len(y_post),) - 1))

    plt.plot(x_pre, y_pre*fps, 'o', alpha=.40, zorder=1, ms=8, mew=1)
    plt.plot(x_post, y_post*fps, 'o', alpha=.40, zorder=1, ms=8, mew=1)

    for i in range(len(x_pre)):
        plt.plot([x_pre, x_post], [y_pre*fps, y_post*fps], color = 'grey', linewidth = 0.5, linestyle = '--', zorder=-1)

    ax = plt.gca()
    ax.set_xticks(range(2))
    ax.set_xticklabels(['Pre', 'Post'])
    ax.set_ylabel('TDA (Degrees/s)')
    plt.tight_layout()
    #if len(figname) == 0:
    plt.show()
    #else:
    #    plt.savefig(figname)


def pre_vs_post_stats(df_data, treatment_grp, stat, metric):

    treat_data = df_data.loc[df_data['Treatment']==treatment_grp]
    pre_data = treat_data.loc[treat_data['Group']=='Pre']
    post_data = treat_data.loc[treat_data['Group']=='Post']

    uID_pre = list(pre_data['Subject ID'].unique())
    uID_post = list(post_data['Subject ID'].unique())
    
    #Only consider subject IDs that are present in both pre and post
    common_uID = list(set(uID_post).intersection(set(uID_pre)))
    
    res = np.zeros((len(common_uID),2))
    for i in range(len(common_uID)):

        df_subid_pre = pre_data.loc[pre_data['Subject ID']==common_uID[i]]
        df_subid_post = post_data.loc[post_data['Subject ID']==common_uID[i]]

        task_list_pre = list(df_subid_pre['Task'])
        task_list_post = list(df_subid_post['Task'])

        #Only consider tasks that are present in both pre and post
        common_tasks = list(set(task_list_post).intersection(set(task_list_pre)))
        
        df_subid_pre = df_subid_pre.loc[df_subid_pre['Task'].isin(common_tasks)]
        df_subid_post = df_subid_post.loc[df_subid_post['Task'].isin(common_tasks)]
        res[i, 0] = np.median(df_subid_pre[metric].values)
        res[i, 1] = np.median(df_subid_post[metric].values)
        #import pdb; pdb.set_trace()        
    
    ad_scale_val = 2.75
    pre_valid_idx = mad_outlier_screen(res[:,0], ad_scale=ad_scale_val)
    post_valid_idx = mad_outlier_screen(res[:,1], ad_scale=ad_scale_val)
    pre_screened = res[pre_valid_idx & post_valid_idx,0] * (180/np.pi)
    post_screened = res[pre_valid_idx & post_valid_idx,1]  * (180/np.pi)
    plot_pre_post_jitter(pre_screened, post_screened, 'temp.svg')
    print('Avg Pre: ', np.mean(pre_screened), 'Avg Post: ', np.mean(post_screened))
    print(stat(pre_screened,post_screened))


def pre_process_data(data_filename, treatment_info_filename, task_list=[]):

    #Load data and append column with treatment info to dataframe
    df_data = load_data_with_treatment_col(data_filename, treatment_info_filename)

    #Make pre/post subject names match by removing unnecessary trailing 7 characters  
    df_data['Subject ID'] = df_data['Subject ID'].apply(lambda x: x.replace(" ", ""))
    df_data['Subject ID'] = df_data['Subject ID'].apply(lambda x: x[:-7])

    return df_data


Datadir = 'Datafiles/'
#fn = "AHA_2015-18_filtered_01302024_v2.csv"
#fn = 'AHA_2015-18_filtered_04042024_v2.csv'
fn = 'AHA_2015-18_filtered_04172024_TotalAngularDistance.csv'
treatfn = 'Intervention_type.csv'
task_list = generate_task_list([3, 4, 5, 6, 7, 8, 9, 11, 12])

df_data = pre_process_data(fn, treatfn)
elb_flex_mean = (df_data['Elbow Flexion L'] + df_data['Elbow Flexion R'])/2.0
df_data = df_data.assign(ElbowFlexion=elb_flex_mean)

#metric = 'Trunk Angular Displacement'
metric = 'ElbowFlexion'
#import pdb; pdb.set_trace()
#df_data = select_from_tasks(df_data, task_list)
#pre_vs_post_stats(df_data, treatment_grp=1, stat=ttest_rel, metric=metric)
pre_vs_post_stats(df_data, treatment_grp=2, stat=ttest_rel, metric=metric)


cmb_opt = False#True

