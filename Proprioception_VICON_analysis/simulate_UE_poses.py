import sys
sys.path.insert(1, '../PEACK_API')
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from PEACKMetrics import graphs, Proprioception
from scipy import stats
import seaborn as sns

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

    #np.random.seed(324)
    pts = np.random.randn(N, 3)
    w = np.linalg.norm(pts, axis=-1)
    w = w.reshape(-1,1)
    pts = pts/w
    return pts

def gen_UE_Pose(Chest, RShldr, LShldr, RElbow, LElbow, RWrist, LWrist):
    Body = {
                'LShoulder':LShldr,
                'RShoulder':RShldr,
                'LElbow':LElbow,
                'RElbow':RElbow,
                'LWrist':LWrist,
                'RWrist':RWrist,
                'Chest':Chest
          }
    return Body


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
        Body = gen_UE_Pose(Chest, RShldr, LShldr, RElbow, LElbow, RWrist, LWrist)
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

    #graphs.plot3Dline(ax,body['RElbow'],body['LElbow'],0,plotmode = 'default', color = 'red')
    #graphs.plot3Dline(ax,body['RWrist'],body['LWrist'],0,plotmode = 'default', color = 'red')

    ax.set_ylim(-2, 2)  #X axis
    ax.set_xlim(-2, 2)
    ax.set_zlim(-2, 2)  #Y axis

    plt.show()


def make_and_assess_custom_demo_pose():

    LShldr = np.array([-1,0,0]).reshape(1,-1)
    RShldr = np.array([1,0,0]).reshape(1,-1)
    Chest = np.array([0,-0.5,0]).reshape(1,-1)

    LElbow = np.array([-2,0,0]).reshape(1,-1)
    RElbow = np.array([1,0,1]).reshape(1,-1)

    LWrist = np.array([-2,1,0]).reshape(1,-1)
    RWrist = np.array([1,1,1]).reshape(1,-1)
    Body = gen_UE_Pose(Chest, RShldr, LShldr, RElbow, LElbow, RWrist, LWrist)
    
    s0 = Proprioception.orientation_symmetry_metric(Body)
    s1 = Proprioception.angle_symmetry_metric(Body)
    s2 = Proprioception.distance_symmetry_metric(Body)
    
    print(np.array([s0, s1, s2]))
    plot_pose(Body)

def find_demo_poses(syms, Poses):

    hOrientSym = [1,0,0]
    hAngSym = [0,1,0]
    hDistSym = [0,0,1]
    hOrientAngSym = [1, 1,0]
    hOrientDistSym = [1,0,1]
    hAngDistSym = [0,1,1]
    perfSym = [1,1,1]

    Targets = np.array([hOrientSym, hAngSym, hDistSym, hOrientAngSym, hOrientDistSym, hAngDistSym, perfSym])

    #import pdb; pdb.set_trace()
    for i in range(1,2):#range(0, len(Targets)):

        sumscores = np.sum(np.abs(syms - Targets[i]),axis=-1)
        target = np.argmin(sumscores)

        print('Symmetries [Orientation, Angle, Distance]:', syms[target,:])
        plot_pose(Poses[target])


N_poses = 10000
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

#import pdb; pdb.set_trace()

X = syms[:,0]; Y = syms[:,1]; Z = syms[:,2]
print(stats.ttest_ind(X,Y))
print(stats.ttest_ind(X,Z))
print(stats.ttest_ind(Y,Z))

print(stats.pearsonr(X,Y))
print(stats.pearsonr(X,Z))
print(stats.pearsonr(Y,Z))

print(stats.spearmanr(X,Y))
print(stats.spearmanr(X,Z))
print(stats.spearmanr(Y,Z))

# X = X[::10]
# Y = Y[::10]
# Z = Z[::10]

# plt.figure()
# sns.regplot(x=X,y=Y)
# plt.figure()
# sns.regplot(x=X,y=Z)
# plt.figure()
# sns.regplot(x=Y,y=Z)
# plt.show()
#make_and_assess_custom_demo_pose()
#find_demo_poses(syms, Poses)
#import pdb; pdb.set_trace()


#Poses[4908] - High angle, Other symmetries low
#Poses[4695] - High distance, low angle symmetries.
#Poses[7756] - High orientation, other symmetries low 
#Poses[4005] - Low orienntation, other symmetries high
