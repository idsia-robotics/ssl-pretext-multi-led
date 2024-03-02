import numpy as np
from scipy.spatial.transform import Rotation as R

def make_homogeneous_mat(m):
    res = np.eye(4)
    res[:-1, :-1] = m
    return res

def make_homogeneous_vec(v):
    in_shape = list(v.shape)
    out_shape = in_shape.copy()
    out_shape[-1] += 1
    res = np.ones(out_shape)
    res[:, :-1] = v
    return res


"""
Camera to base link transform as per tf2

translation=geometry_msgs.msg.Vector3(x=0.07452833958517707, y=0.0006071374557668694, z=0.23504998530719307)
rotation=geometry_msgs.msg.Quaternion(x=-0.49034123379425376, y=0.4886326003895837, z=-0.5094044131423495, w=0.5111856806961056)
"""

base_to_camera = np.array([
    [ 1.11022302e-16, -1.00000000e+00,  0.00000000e+00,  3.40000000e-04],
    [ 4.50946877e-02,  1.11022302e-16, -9.98982717e-01,  2.31722261e-01],
    [ 9.98982717e-01,  0.00000000e+00,  4.50946877e-02, -8.49624244e-02],
    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]
)

gimbal_to_base = np.array([
    [ 1.,     0.,     0.,    -0.002],
    [ 0.,     1.,     0.,     0.   ],
    [ 0.,     0.,     1.,     0.118],
    [ 0.,     0.,     0.,     1.   ],
])

camera_to_base = np.linalg.inv(base_to_camera)

image_to_base = camera_to_base


# P = np.array([
#             [265.423279, 0.0 ,317.281071,0.0,],
#             [0.0,265.08313,178.666968,0.0],
#             [0.0,0.0,1.0,0.0]
#         ])


K = np.array([305.86615474,   0.0, 313.32092337,  0.0, 307.25209474, 183.31037687,  0.0,0.0,   1.0]).reshape(3,3)
P = np.zeros((3, 4))
P[:, :-1] = K

# K = np.array([
#     [266.082019,0.0,317.509713],
#     [0.0,265.623549,179.159722],
#     [0, 0, 1]
#     ])

d = np.array([-0.0854973,   0.04988998, -0.0031321,  -0.00019603, -0.04222285])

def reconstruct_position(proj_uv, x_coord):
    """
    Models predict the image-space position of the
    gimbal joint on the robomaster.
    They also predict the x coordinate of the visible robot
    relative to the frame of reference of the base joint of
    the camera robot.

    This function first computes the predicted position of the gimbal
    in the frame of reference of the base joint. Next, this position
    is recalibrated to point to the base joint of the visible robot.
    """
    
    # proj_uv = proj_uv - np.array([[320, 170]])

    # backproj_cv = cv2.undistortPoints(
    #     proj_uv,
    #     cameraMatrix=K,
    #     distCoeffs=d,
    # )[0, :].T

    # backproj_cv = np.concatenate(
    #     (
    #         backproj_cv,
    #         np.ones((2,backproj_cv.shape[-1])),
    #     ), axis = 0)

    proj_uv_homo = np.ones((3, proj_uv.shape[1]))
    proj_uv_homo[:-1, :] = proj_uv
    # proj_uv_homo[1, :] = 360 - proj_uv_homo[1, :]

    backproj = np.linalg.pinv(P) @ proj_uv_homo
    backproj[-1, :] = 1
    ray_to_gimbal = (image_to_base @ backproj)
    reconstructed_position = ray_to_gimbal * (x_coord / ray_to_gimbal[0, :])[None, ...]
    reconstructed_position[2, :] = 0
    reconstructed_position[0, :] -= -0.002
    
    # reconstructed_position /= reconstructed_position[-1, :]
    

    # ray_to_base = (ray_to_gimbal @ gimbal_to_base).T
    # ray_to_base /= ray_to_base[-1, :]
    return reconstructed_position[:-1, ...]





    # Backproject


    # Camera -> Base

    # Add orientation
    
