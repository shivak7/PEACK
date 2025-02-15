import numpy as np
from PEACKMetrics.metrics import angle_between_ts_vectors
from matplotlib import pyplot as plt


def angular_velocity(Body):

    rankle = Body['RAnkle']
    lankle = Body['LAnkle']
    side = []
    if len(rankle) < 1:
        side = 'L'
    elif len(lankle) < 1:
        side = 'R'
    else:
        lsig = np.std(lankle)
        rsig = np.std(rankle)
        side = 'R'
        if lsig > rsig:
            side = 'L'

    ankle_label = side + 'Ankle'
    knee_label = side + 'Knee'
    hip_label = side + 'Hip'

    ankle = Body[ankle_label]
    knee = Body[knee_label]
    hip = Body[hip_label]

    vector_knee_hip = hip - knee
    vector_knee_ankle = ankle - knee

    theta = 180*angle_between_ts_vectors(vector_knee_hip,vector_knee_ankle)/np.pi

    plt.plot(Body.time/1000, theta)
    plt.ylabel('Angle (deg)')
    plt.xlabel('Time (s)')
    plt.show()


    import pdb; pdb.set_trace()