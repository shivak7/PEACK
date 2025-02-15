import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from PreProcess import PreProcess

def LoadViconFile(fn):

    temp = pd.read_csv(fn,header=None, skiprows=5).values
    data = PreProcess(temp[:, 2:], '3D', drop_cols = 0, do_interp=False); del temp
    MedPosns = np.nanmedian(data, axis=0)
    Njoints = int(len(MedPosns)/3)
    X = MedPosns.reshape((Njoints,3))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    a1 = ax.plot(X[:,0],X[:,1],X[:,2], linestyle='None', marker='o', color='b')
    ax.set_xlabel('Z (m)')
    ax.set_ylabel('X (m)')
    ax.set_zlabel('Y (m)')
    plt.show()


    import pdb; pdb.set_trace()
