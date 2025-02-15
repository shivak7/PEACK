import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



def plot_task(df, Yval = "Trunk Rotation Angle", ylim = 40):
    sns.set_theme(style="darkgrid", font_scale=2)
    plt.figure()
    #"Trunk Sway Distance"#"Trunk Rotation Angle"
    sns.boxplot(data=df, x="Group", y=Yval, hue="Task",
               linewidth=1, palette="vlag", showfliers = False)
    sns.stripplot(data=df, x="Group", y=Yval, hue="Task", dodge=True)
    sns.despine(left=True)
    plt.ylim(bottom = 0, top = ylim)
    plt.legend([],[], frameon=False)
    plt.show()

df = pd.read_csv("AHA_2015_2017_2018_filtered.csv")
#df_task1 = df.loc[df['Task'] == "Task 1"]
#df_powerbars = df.loc[df['Pose'] == "PowerBars"]

plot_task(df, "Trunk Rotation Angle", 60)
plot_task(df, "Trunk Sway Angle", 0.6)
plot_task(df, "Trunk Sway Distance", 25)

#plot_task(df_powerbars)

# df_mus_dist = df_muscles.loc[df_muscles['Metric']=='Distance']
# df_mus_dist_CP_AH = df_mus_dist[df_mus_dist['Group']=='CP AH']
# df_mus_dist_TD = df_mus_dist.loc[df_mus_dist['Group']=='TD Control']
# df_mus_dist_Adult = df_mus_dist.loc[df_mus_dist['Group']=='Adult Control']
#
# df_mus_dist_CP.describe()
# df_mus_dist_TD.describe()
# df_mus_dist_Adult.describe()
# import pdb; pdb.set_trace()
