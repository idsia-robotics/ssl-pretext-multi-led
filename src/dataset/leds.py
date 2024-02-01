from src.inference import P, gimbal_to_base, base_to_camera
import numpy as np

LED_BB_TO_BASE_LINK = np.array([
    [-1.0000000e+00, -1.2246468e-16,  0.0000000e+00, -1.6692550e-01],
    [ 1.2246468e-16, -1.0000000e+00,  0.0000000e+00, -1.5023000e-03],
    [ 0.0000000e+00,  0.0000000e+00,  1.0000000e+00,  6.5054300e-02],
    [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00],
])

LED_BF_TO_BASE_LINK = np.array([
    [ 1.,         0.,         0.,         0.1657413],
    [ 0.,         1.,         0.,        -0.0017117],
    [ 0.,         0.,         1.,         0.0779737],
    [ 0.,         0.,         0.,         1.       ],
])

LED_BL_TO_BASE_LINK = np.array([
    [ 2.22044605e-16, -1.00000000e+00,  0.00000000e+00, -1.15720000e-03],
    [ 1.00000000e+00,  2.22044605e-16,  0.00000000e+00,  9.74360000e-02],
    [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00,  6.45216000e-02],
    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]],
)

LED_BR_TO_BASE_LINK = np.array([
    [ 2.22044605e-16, -1.00000000e+00,  0.00000000e+00, -1.15720000e-03],
    [ 1.00000000e+00,  2.22044605e-16,  0.00000000e+00,  -9.74360000e-02],
    [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00,  6.45216000e-02],
    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]],
)

LED_TR_TO_BASE_LINK = np.array([
    [ 0.99999848,  0.00174533,  0.,         -0.00180114],
    [-0.00174533,  0.99999848, -0.,         -0.06941045],
    [-0.        ,  0.,          1.,          0.20561   ],
    [ 0.        ,  0.,          0.,          1.        ]
])

LED_TL_TO_BASE_LINK = np.array([
    [ 0.99999848,  0.00174533,  0.,         -0.00180114],
    [-0.00174533,  0.99999848, -0.,         0.06941045],
    [-0.        ,  0.,          1.,          0.20561   ],
    [ 0.        ,  0.,          0.,          1.        ]
])


_transform_stack = np.stack([
    LED_BB_TO_BASE_LINK,
    LED_BL_TO_BASE_LINK,
    LED_BR_TO_BASE_LINK,
    LED_BF_TO_BASE_LINK,
    LED_TL_TO_BASE_LINK,
    LED_TR_TO_BASE_LINK
], axis = 0)

LED_VISIBILITY_RANGES_DEG = [
        [[-180, -100], [100, 180]],
        [[35, 145], [np.inf, np.inf]],
        [[-145, -35], [np.inf, np.inf]],
        [[-80, 80], [np.inf, np.inf]],
        [[0, 180], [np.inf, np.inf]],
        [[-180, -0], [np.inf, np.inf]],
    ]

LED_VISIBILITY_RANGES_RAD = np.deg2rad(LED_VISIBILITY_RANGES_DEG)

LED_TYPES = ["bb", "bl", "br", "bf", "tl", "tr"]


def compute_led_visibility(target_to_cam_robot, cam_robot_position_rel, image_boundaries = (640, 360)):
    leds_to_robot_base_link = target_to_cam_robot @ _transform_stack[:, :, -1].T
    led_to_robot_camera = base_to_camera @ leds_to_robot_base_link
    leds_proj = (P @ led_to_robot_camera).T
    proj_cond = ((leds_proj[:, 0] <= image_boundaries[0]) & (leds_proj[:, 0] >= 0)) &\
                ((leds_proj[:, 1] <= image_boundaries[1]) & (leds_proj[:, 1] >= 0))
    
    other_theta_rel = np.arctan2(cam_robot_position_rel[1], cam_robot_position_rel[0])
    led_visibility = (other_theta_rel >= LED_VISIBILITY_RANGES_RAD[:, :, 0]) &\
            (other_theta_rel <= LED_VISIBILITY_RANGES_RAD[:, :, 1])
    ori_cond = np.logical_or.reduce(led_visibility, axis = 1)
    return proj_cond & ori_cond

    



