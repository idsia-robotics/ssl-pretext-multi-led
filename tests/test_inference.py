import unittest

import numpy as np
from src.inference import d, K, reconstruct_position, base_to_camera, P
import cv2
from scipy.spatial.transform import Rotation as R


class TestBackprojection(unittest.TestCase):

    def test_backprojection(self):
        true_position = np.array([[ 1.4439,  0.3092, .23, 1]]).T # Base link

        proj = P @ base_to_camera @ true_position
        
        # cv_proj = cv2.projectPoints(
        #     true_position[:-1, :],
        #     tvec=base_to_camera[:-1, -1],
        #     rvec=R.from_matrix(base_to_camera[:-1, :-1]).as_rotvec(),
        #     distCoeffs=d,
        #     cameraMatrix=K

        # )[0][0, :].T
        # cv_proj = np.concatenate((cv_proj, np.ones((1,cv_proj.shape[-1])).T), axis = 0)
        proj /= proj[-1]

        back = reconstruct_position(proj[:-1, :], np.array([[1.4439]]))
        self.assertAlmostEqual(
            np.linalg.norm(true_position[:-1, :]),
            np.linalg.norm(back),
        )


