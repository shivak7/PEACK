import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

fig, axs = plt.subplots(3,3)
#Title = ['Unimanual Task', 'Bimanual Task', 'Unimanual Task', 'Bimanual Task']
vidx = [0, 4, 1, 5, 2, 6, 3, 7]

for i in range(8):

    pfilename = 'Pfile' + str(i) + '.txt'
    vfilename = 'Vfile' + str(vidx[i]) + '.txt'
    Pdata = np.loadtxt(pfilename)
    Vdata = np.loadtxt(vfilename)

    Ptime = Pdata[0]
    Ptime = Ptime - np.min(Ptime)
    #Ptime = Ptime/1000
    Pval = Pdata[1]

    Vtime = Vdata[0]
    Vtime = Vtime - np.min(Vtime)
    Vval = Vdata[1]
    
    ax = plt.subplot(3,3,i+1)
    #ax.title.set_text(Title[i])
    plt.plot(Ptime, Pval)
    plt.plot(Vtime, Vval)
    plt.xlabel('Time (s)')
    plt.ylabel('TDA (radians)')
    plt.legend(['OpenPose', 'VICON'])

    # #import pdb; pdb.set_trace()
    # T = np.arange(Vtime[0],Vtime[-1], np.mean(np.diff(Vtime)))
    # T2 = np.arange(Vtime[0],Vtime[-1], np.mean(np.diff(Ptime)))
    
    # interpolated = interp1d(T, Vval[:-1], kind='cubic')
    
    # Vval_interp = interpolated(T2)
    
    # corr = np.correlate(Pval - np.mean(Pval), Vval_interp - np.mean(Vval_interp), mode='full')
    # lag = corr.argmax() - (len(Pval) - 1)
    # #import pdb; pdb.set_trace()
    
    # ax = plt.subplot(2,2,i+1)
    # ax.title.set_text(Title[i])
    # plt.plot(Ptime, Pval)
    # plt.plot(T2, Vval_interp)
    # plt.xlabel('Time (s)')
    # plt.ylabel('TDA (radians)')
    # plt.legend(['OpenPose', 'VICON'])
plt.show()

P_uni = [0.022487934, 0.030938221, 0.026346563, 0.022228314]
P_bim = [0.061328006, 0.056003848, 0.066780589, 0.047638237]

V_uni = [0.042641372, 0.040002901, 0.058418116, 0.028110937]
V_bim = [0.092741587, 0.05464793, 0.090753788, 0.033803978]

P_all = np.hstack((P_uni, P_bim))
V_all = np.hstack((V_uni, V_bim))

print('Correlation between the VICON and PEACK TDA: ', np.corrcoef(P_all, V_all)[0,1])
#import pdb; pdb.set_trace()
