import numpy as np


###Powerbars Validation Template

# LShoulder = []
# LElbow = []
# LWrist = []

# RShoulder = []
# RLElbow = []
# RWrist = []

###


def joint_angle_in_degrees(vertex1, vertex2, vertex3):

    limb_vector_1 = np.array(vertex1) - np.array(vertex2)
    limb_vector_2 = np.array(vertex3) - np.array(vertex2)
    dotp = np.dot(limb_vector_1, limb_vector_2)
    mag = np.linalg.norm(limb_vector_1)*np.linalg.norm(limb_vector_2)
    angle = np.arccos(dotp/mag)*(180/np.pi)

    return angle

# TDPowerbars Participant 17 (Vicon marker identification)

# LShoulder = [0.073,0.225, 1.938]
# LElbow = [-0.186, 0.27, 1.997]
# LWrist = [-0.192, 0.454, 1.911]

# RShoulder = [0.267,0.235, 1.916]
# RElbow = [0.509, 0.26, 1.831]
# RWrist = [0.456, 0.433, 1.748]

# TDPowerbars Participant 17 (Vicon marker identification)

LShoulder = [0.031,0.276, 1.96]
LElbow = [-0.165, 0.291, 2.01]
LWrist = [-0.217, 0.443, 1.916]

RShoulder = [0.303,0.276, 1.895]
RElbow = [0.5, 0.283, 1.824]
RWrist = [0.478, 0.437, 1.728]


LShoulder_Angle = joint_angle_in_degrees(RShoulder, LShoulder, LElbow)
RShoulder_Angle = joint_angle_in_degrees(LShoulder, RShoulder, RElbow)

LElbow_Angle = joint_angle_in_degrees(LShoulder, LElbow, LWrist)
RElbow_Angle = joint_angle_in_degrees(RShoulder, RElbow, RWrist)

print('Shoulder Angle: Left vs Right ', LShoulder_Angle, '\t', RShoulder_Angle)
print('Elbow Angle: Left vs Right ', LElbow_Angle, '\t', RElbow_Angle)