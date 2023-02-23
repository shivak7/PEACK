import sys
sys.path.insert(1, '/Users/shiva/Dropbox/Burke Work/DeepMarker/Processed Data/PythonScripts/PEACK_API')
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from PEACKMetrics import graphs, Proprioception

def plot_sphere(x,y,z):

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.plot(x,y,z,'r.')
    plt.show()


def unit_sphere():
    '''
    Unit sphere:
    Eqn: x^2 + y^2 + z^2 = 1

    '''

    x = np.arange(-1,1.1,0.01)
    y = np.arange(-1,1.1,0.01)
    xx, yy = np.meshgrid(x,y)
    xx = xx.flatten()
    yy = yy.flatten()
    zz = np.sqrt(1 - (xx**2 + yy**2))
    zflip = -zz[1::2]
    zz[1::2] = zflip
    return [xx,yy,zz]

def unit_sphere_random_points(N=1000):

    pts = np.random.randn(N, 3)
    w = np.linalg.norm(pts, axis=-1)
    w = w.reshape(-1,1)
    pts = pts/w
    return pts

def gen_random_UE_poses(nPoses=10):

    pts = unit_sphere_random_points(100000)
    L = len(pts)
    Poses = []
    for i in range(nPoses):
        LShldr = np.array([-1,0,0]).reshape(1,-1)
        RShldr = np.array([1,0,0]).reshape(1,-1)
        LElbow = LShldr + pts[np.random.randint(L,size=1)].reshape(1,-1)
        LWrist = LElbow + pts[np.random.randint(L,size=1)].reshape(1,-1)
        RElbow = RShldr + pts[np.random.randint(L,size=1)].reshape(1,-1)
        RWrist = RElbow + pts[np.random.randint(L,size=1)].reshape(1,-1)
        Body = {
                'LShldr':LShldr,
                'RShldr':RShldr,
                'LElbow':LElbow,
                'RElbow':RElbow,
                'LWrist':LWrist,
                'RWrist':RWrist
          }
        Poses.append(Body)
    return Poses
#plot_sphere(pts[:,0], pts[:,1], pts[:,2])

def plot_pose(body):
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    graphs.plot3Dline(ax,body['LShldr'],body['RShldr'],0,plotmode = 'default', color = 'gray')
    graphs.plot3Dline(ax,body['LShldr'],body['LElbow'],0,plotmode = 'default', color = 'gray')
    graphs.plot3Dline(ax,body['LElbow'],body['LWrist'],0,plotmode = 'default', color = 'gray')
    graphs.plot3Dline(ax,body['RShldr'],body['RElbow'],0,plotmode = 'default', color = 'gray')
    graphs.plot3Dline(ax,body['RElbow'],body['RWrist'],0,plotmode = 'default', color = 'gray')
    plt.show()


Poses = gen_random_UE_poses(1000)
# for i in range(10):
#     plot_pose(Poses[i])
