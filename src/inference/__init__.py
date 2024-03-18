import cv2
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

# base_to_camera = np.array([
#     [ 1.11022302e-16, -1.00000000e+00,  0.00000000e+00,  3.40000000e-04],
#     [ 4.50946877e-02,  1.11022302e-16, -9.98982717e-01,  2.31722261e-01],
#     [ 9.98982717e-01,  0.00000000e+00,  4.50946877e-02, -8.49624244e-02],
#     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]
# )
camera_to_base = np.array([
[-3.49065734e-03,  2.76530096e-02,  9.99611488e-01 , 7.49351087e-02],
[-9.99993908e-01, -9.65389548e-05, -3.48932213e-03 , 7.14467891e-05],
[ 1.11899537e-08, -9.99617578e-01,  2.76531781e-02 , 2.33974814e-01],
[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00 , 1.00000000e+00],
])
image_to_base = camera_to_base


gimbal_to_base = np.array([
    [ 1.,     0.,     0.,    -0.002],
    [ 0.,     1.,     0.,     0.   ],
    [ 0.,     0.,     1.,     0.118],
    [ 0.,     0.,     0.,     1.   ],
])

base_to_camera = np.linalg.inv(camera_to_base)



# P = np.array([
#             [265.423279, 0.0 ,317.281071,0.0,],
#             [0.0,265.08313,178.666968,0.0],
#             [0.0,0.0,1.0,0.0]
#         ])


K = np.array([314.26712497,   0.0, 322.317073, 0.0, 314.24818945, 176.7395928, 0.0, 0.0, 1.0]).reshape(3,3)
P = np.zeros((3, 4))
P[:, :-1] = K

# K = np.array([
#     [266.082019,0.0,317.509713],
#     [0.0,265.623549,179.159722],
#     [0, 0, 1]
#     ])

d = np.array([-0.07173926,  0.0384923,  -0.00145677,  0.0002169,  -0.03831285])

def reconstruct_position(proj_uv, x_coord, camera_to_base = image_to_base[None, ...]):
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
    #     distCoeffs=np.zeros((1,4)),
    # ).squeeze().T

    # backproj_cv = np.concatenate(
    #     (
    #         backproj_cv,
    #         np.ones((2,backproj_cv.shape[-1])),
    #     ), axis = 0)
    proj_uv_homo = np.ones((3, proj_uv.shape[-1]))
    proj_uv_homo[:-1, :] = proj_uv
    # proj_uv_homo[1, :] = 360 - proj_uv_homo[1, :]

    backproj = np.linalg.pinv(P) @ proj_uv_homo
    # backproj = backproj_cv
    backproj[-1, :] = 1
    ray_to_gimbal = (camera_to_base @ backproj.T[..., None]).squeeze(-1).T
    reconstructed_position = ray_to_gimbal * (x_coord / ray_to_gimbal[0, :])[None, ...]
    reconstructed_position[2, :] -= .118
    # reconstructed_position[0, :] -= -0.002
    
    # reconstructed_position /= reconstructed_position[-1, :]
    

    # ray_to_base = (ray_to_gimbal @ gimbal_to_base).T
    # ray_to_base /= ray_to_base[-1, :]
    return reconstructed_position[:-1, ...]
    # Backproject


    # Camera -> Base

    # Add orientation
    

def point_rel_to_camera(point, camera_pose = base_to_camera):
    """
    Expects point to be Nx3
    """
    point_homo = np.pad(point, ((0,0), (0,1)), mode='constant', constant_values=1)
    res = cv2.projectPoints((camera_pose @ point_homo.T).T[..., :3].astype(np.float32),
                    np.zeros(3), np.zeros(3),
                    K,
                    np.zeros(0))[0].squeeze()
    return res
