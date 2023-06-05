import sys
sys.path.insert(1, '../PEACK_API')
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

    np.random.seed(324)
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
        Chest = np.array([0,-0.5,0]).reshape(1,-1)
        LElbow = LShldr + pts[np.random.randint(L,size=1)].reshape(1,-1)
        LWrist = LElbow + pts[np.random.randint(L,size=1)].reshape(1,-1)
        RElbow = RShldr + pts[np.random.randint(L,size=1)].reshape(1,-1)
        RWrist = RElbow + pts[np.random.randint(L,size=1)].reshape(1,-1)
        Body = {
                'LShoulder':LShldr,
                'RShoulder':RShldr,
                'LElbow':LElbow,
                'RElbow':RElbow,
                'LWrist':LWrist,
                'RWrist':RWrist,
                'Chest':Chest
          }
        Poses.append(Body)
    return Poses
#plot_sphere(pts[:,0], pts[:,1], pts[:,2])

def plot_pose(body):
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    graphs.plot3Dline(ax,body['LShoulder'],body['RShoulder'],0,plotmode = 'default', color = 'gray')
    graphs.plot3Dline(ax,body['LShoulder'],body['LElbow'],0,plotmode = 'default', color = 'gray')
    graphs.plot3Dline(ax,body['LElbow'],body['LWrist'],0,plotmode = 'default', color = 'gray')
    graphs.plot3Dline(ax,body['RShoulder'],body['RElbow'],0,plotmode = 'default', color = 'gray')
    graphs.plot3Dline(ax,body['RElbow'],body['RWrist'],0,plotmode = 'default', color = 'gray')
    graphs.plot3Dline(ax,(body['RShoulder'] + body['LShoulder'])/2.0,body['Chest'],0,plotmode = 'default', color = 'gray')

    graphs.plot3Dline(ax,body['RElbow'],body['LElbow'],0,plotmode = 'default', color = 'red')
    graphs.plot3Dline(ax,body['RWrist'],body['LWrist'],0,plotmode = 'default', color = 'red')

    ax.set_ylim(-2, 2)  #X axis
    ax.set_xlim(-2, 2)
    ax.set_zlim(-2, 2)  #Y axis

    plt.show()


N_poses = 1000
Poses = gen_random_UE_poses(N_poses)
#plot_pose(Poses[0])
#o_sym = Proprioception.orientation_symmetry_metric(Poses[617])                      #748- min, 617 max for random seed 0
# o_sym = Proprioception.orientation_symmetry_metric(Poses[0])
# a_sym = Proprioception.angle_symmetry_metric(Poses[0])
# d_sym = Proprioception.distance_symmetry_metric(Poses[0])
#
# #print(o_sym, a_sym, d_sym)

syms = np.zeros((N_poses, 3))
for i in range(N_poses):
    syms[i][0] = Proprioception.orientation_symmetry_metric(Poses[i])
    syms[i][1] = Proprioception.angle_symmetry_metric(Poses[i])
    syms[i][2] = Proprioception.distance_symmetry_metric(Poses[i])

import pdb; pdb.set_trace()
