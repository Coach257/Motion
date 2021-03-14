import json

import numpy as np
from scipy.spatial.transform import Rotation as R

from .vector_util import vector_mod
from .coordinates_transform import SkeletonCoordinatesTransform


class FootContact(object):
    def __init__(self, joints, vel_threshold):
        super().__init__()
        self.joints = joints
        self.vel_threshold = vel_threshold

    def judge_by_vel(self, position):
        assert len(position.shape) == 3 and position.shape[-2] == len(self.joints) and position.shape[0] > 1
        l_ank, r_ank, l_foot, r_foot = self.joints.index("L_Ankle"), self.joints.index("R_Ankle"), self.joints.index("L_Foot"), self.joints.index("R_Foot")
        position = position[:, [l_ank, r_ank, l_foot, r_foot]]
        vel = position[1:] - position[:-1]
        vel = np.concatenate((vel, vel[-1:]), axis=0)
        # import ipdb; ipdb.set_trace()
        return vector_mod(vel) < self.vel_threshold


if __name__ == "__main__":
    joint_info = json.load(open('/home/data/Motion3D/AMASS/joint_info.json'))
    sct = SkeletonCoordinatesTransform(joint_info["joints"], joint_info["parent"], joint_info["skel_offset"])
    
    data = np.load("/home/data/Motion3D/AMASS/20160330_03333/walking_poses.npz")
    rotation = data["poses"].reshape(-1, 52, 3)[:, :24].reshape(-1, 3)
    rotation = R.from_rotvec(rotation).as_quat().reshape(-1, 24, 4)
    skel_global_position = sct.forward_kinematics(rotation, data["trans"], rot_type="local")

    foot_contact = FootContact(joint_info["joints"], 1e-4)
    foot_contact.judge_by_vel(skel_global_position)