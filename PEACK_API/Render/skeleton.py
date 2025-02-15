from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import matplotlib.animation as animation

def data_for_cylinder_along_z(c0,c1, r):
    
    
    z = np.linspace(c0[2], c1[2], 10)
    theta = np.linspace(0, 2*np.pi, 50)
    theta_grid, z_grid=np.meshgrid(theta, z)
    x_grid = r*np.cos(theta_grid) + center_x
    y_grid = r*np.sin(theta_grid) + center_y
    return x_grid,y_grid,z_grid


def plot_segment(l1,l2, ax, t):

    z_buff = np.zeros((len(l1),1))
    l1 = np.hstack((l1, z_buff))
    l2 = np.hstack((l2, z_buff))

    points = np.vstack((l1[t],l2[t]))

    ax.plot(points[:,0], points[:,1], points[:,2])

def plot_body(t, Body, ax):
    ax.clear() 
    plot_segment(Body['Neck'], Body['RShoulder'], ax, t)    
    plot_segment(Body['Neck'], Body['LShoulder'], ax, t)
    plot_segment(Body['RShoulder'], Body['LShoulder'], ax, t)
    plot_segment(Body['Neck'], Body['MidHip'], ax, t)

    plot_segment(Body['LElbow'], Body['LShoulder'], ax, t)
    plot_segment(Body['LElbow'], Body['LWrist'], ax, t)

    plot_segment(Body['RElbow'], Body['RShoulder'], ax, t)
    plot_segment(Body['RElbow'], Body['RWrist'], ax, t)

    #line = ax.plot([],[], 'o', c='blue', lw=1)
    #return line


def animate_body(Body):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=-90, azim=-90, roll=0)
    
    ani = animation.FuncAnimation(fig, plot_body, len(Body.time), fargs=(Body, ax), blit=False,interval=20)

    plt.show()
    import pdb; pdb.set_trace()


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# Xc,Yc,Zc = data_for_cylinder_along_z(0.2,0.2,0.05,0.1)
# ax.plot_surface(Xc, Yc, Zc, alpha=0.5)

# plt.show()