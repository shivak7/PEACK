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


def plot_scores(df_data, metric):

    sns.set_style('darkgrid')

    g = sns.barplot(df_data, x='Task', y=metric, hue='Group')
    plt.show()

def plot_scores2(df_data_grouped_by_time, metric):

    sns.set_style('darkgrid')

    g = sns.barplot(df_data_grouped_by_time, x='Task', y=metric, hue='Treatment')
    plt.show()
    
def pre_vs_post_stats(df_data, treatment_grp, stat, combine_tasks=True):

    if combine_tasks == True:
        df_data = combine_across_tasks(df_data)

    metric = 'Trunk Angular Displacement'
    treat_data = df_data.loc[df_data['Treatment']==treatment_grp]
    pre_data = treat_data.loc[treat_data['Group']=='Pre']
    post_data = treat_data.loc[treat_data['Group']=='Post']
    x = pre_data[metric].values
    y = post_data[metric].values
    #plot_scores(treat_data, metric)

    if(len(x) < len(y)):

        no_missing = len(y) - len(x)
        missing = np.median(x) * np.ones(no_missing,)
        x = np.hstack((x,missing))
    elif (len(x) > len(y)):
        no_missing = len(x) - len(y)
        missing = np.median(y) * np.ones(no_missing,)
        y = np.hstack((y,missing))

    #import pdb; pdb.set_trace()
    print(stat(x,y))

def mad_outlier_screen(data_array, ad_scale=3):             #Median abs deviation based screening

    ad = np.nanmean(np.absolute(data_array - np.nanmean(data_array))) #absolute
    ub = np.nanmedian(data_array) + ad_scale*ad
    lb = np.nanmedian(data_array) - ad_scale*ad
    
    #import pdb; pdb.set_trace()
    valid_data_idx = (data_array < ub) & (data_array > lb)  #screen for outliers
    return valid_data_idx

def treatment_comp_stats(df_data, time_group, stat, combine_tasks=True):

    if combine_tasks == True:
        df_data = combine_across_tasks(df_data)

    metric = 'Trunk Angular Displacement'
    time_group_data = df_data.loc[df_data['Group']==time_group]
    treatment_group_1 = time_group_data.loc[time_group_data['Treatment']==1]
    treatment_group_2 = time_group_data.loc[time_group_data['Treatment']==2]
    x = treatment_group_1[metric].values
    y = treatment_group_2[metric].values
    plot_scores2(time_group_data, metric)
    #import pdb; pdb.set_trace()
    print(stat(x,y))


def plot_outliers(df_data, data_selection_list, data_selection_values, metric, alpha=1):
    
    df_res = df_data.copy()
    for i in range(len(data_selection_list)):
        df_res = df_res.loc[df_res[data_selection_list[i]]==data_selection_values[i]]
    
    Taskvals = df_data['Task'].unique()
    grid_len = int(np.ceil(np.sqrt(len(Taskvals))))

    for j in range(len(Taskvals)):
        
        df_sel = df_res.loc[df_res['Task']==Taskvals[j]]
        np_data = df_sel[metric].values
        plt.subplot(grid_len, grid_len, j+1)
        
        plt.title(Taskvals[j])
        plt.hist(np_data,bins=20, alpha=alpha)
    #plt.show()
    #import pdb; pdb.set_trace()
    #plt.hist(np_data,bins=400, alpha = 0.5)
    

def get_datasubset(df_data, data_selection_list, data_selection_values, metric, by_task = True):

    df_res = df_data.copy()
    for i in range(len(data_selection_list)):
        df_res = df_res.loc[df_res[data_selection_list[i]]==data_selection_values[i]]

    if by_task == True:
        subset_data = []
        Taskvals = df_data['Task'].unique()
        for j in range(len(Taskvals)):
            df_sel = df_res.loc[df_res['Task']==Taskvals[j]]
            subset_data.append(df_sel[metric].values)
        return subset_data
    else:
        return df_res[metric].values

def compare_datasubsets(set1, set2, df_data, data_selection_list, metric, stat):

    set1_data = get_datasubset(df_data, data_selection_list, set1, metric)
    set2_data = get_datasubset(df_data, data_selection_list, set2, metric)

    

    df_combo = combine_across_tasks(df_data)    
    
    set1_combo = get_datasubset(df_combo, data_selection_list, set1, metric, by_task=False)
    set2_combo = get_datasubset(df_combo, data_selection_list, set2, metric, by_task=False)

    if len(set1_data) != len(set2_data):
        print('Error: Mismatch in number of datasubsets within each dataset!')
        return
    
    for i in range(len(set1_data)):

        s1_task = set1_data[i]
        s2_task = set2_data[i]
        print('Sample sizes:', len(s1_task), len(s2_task))
        print('Task', str(i+1), ' :', stat(s1_task, s2_task))

    s1_task_flat = np.hstack(set1_data)
    s2_task_flat = np.hstack(set2_data)
    
    print()
    print('Across all Tasks pooled together:', stat(s1_task_flat, s2_task_flat))
    print('Across Tasks combined: ', stat(set1_combo, set2_combo))
    #import pdb; pdb.set_trace()


Datadir = 'Datafiles/'
#fn = "AHA_2015-18_filtered_01302024_v1.csv"
fn = 'AHA_2015-18_filtered_04042024_v2.csv'
treatfn = 'Intervention_type.csv'
df_data = load_data_with_treatment_col(fn, treatfn)
task_list = generate_task_list([3, 4, 5, 6, 7, 8, 9, 11, 12])
#df_data = select_from_tasks(df_data, task_list)
cmb_opt = False#True

all_data_headers = list(df_data.columns)
data_headers = all_data_headers[1:-3]
print('Available data variables for stats are:')
for i in range(len(data_headers)):
    print(str(i+1)+'. ', data_headers[i])


dataset1 = ['Pre', 1]
dataset2 = ['Pre', 2]
dataset3 = ['Post', 1]
dataset4 = ['Post', 2]

#sel_stat=mannwhitneyu
sel_stat=ttest_ind
#Compare TDA metrics between treatment groups Pre-intervention
#compare_datasubsets(dataset1, dataset2, df_data, data_selection_list= ['Group', 'Treatment'], metric='Trunk Angular Displacement', stat=sel_stat)

#Compare TDA metrics between treatment group 1 Pre vs Post-intervention
#compare_datasubsets(dataset1, dataset3, df_data, data_selection_list= ['Group', 'Treatment'], metric='Trunk Angular Displacement', stat=ttest_rel)

#Compare TDA metrics between treatment group 2 Pre vs Post-intervention
#compare_datasubsets(dataset2, dataset4, df_data, data_selection_list= ['Group', 'Treatment'], metric='Trunk Angular Displacement', stat=ttest_rel)

#plot_outliers(df_data, data_selection_list = ['Group', 'Treatment'], data_selection_values = ['Pre', 1,], metric='Trunk Angular Displacement')
#plot_outliers(df_data, data_selection_list = ['Group', 'Treatment'], data_selection_values = ['Pre', 2,], metric='Trunk Angular Displacement', alpha=0.8)
#plt.show()

# #sel = input('Select index no. of data variable to run stats on. Press q to quit:')
# #print('Selected input:', sel)
# print()
# print('Comparing Pre vs Post')
# pre_vs_post_stats(df_data, treatment_grp=1, stat=ttest_rel, task_list=task_list, combine_tasks=cmb_opt)
# pre_vs_post_stats(df_data, treatment_grp=2, stat=ttest_rel, task_list=task_list, combine_tasks=cmb_opt)
# print()
# print('Comparing treatment groups')
#treatment_comp_stats(df_data, time_group='Pre', stat=ttest_ind, combine_tasks=cmb_opt)
treatment_comp_stats(df_data, time_group='Post', stat=ttest_ind, combine_tasks=cmb_opt)
# #import pdb; pdb.set_trace()