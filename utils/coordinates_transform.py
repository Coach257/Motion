import json

import numpy as np
from scipy.spatial.transform import Rotation as R
from .vector_util import vector_norm
from .pivots import Pivots
from .quaternions import Quaternions


class SkeletonCoordinatesTransform(object):
    """
    :param joints - Joint names
    :param parent - Joint parent index
    """
    def __init__(self, joints, parent, offset):
        self.joints = joints
        self.parent = parent
        self.skel_offset = np.array(offset)
        
        assert len(self.skel_offset.shape) == 2
        for i, p in enumerate(parent):
            assert p < i

    """
    :param rotation - Joint global rotation
    """
    def rotation_global2local(self, rotation):
        assert rotation.shape[-2] == len(self.joints)
        raw_size = rotation.shape
        local_rotation = np.zeros_like(rotation)
        if len(raw_size) == 2:
            rotation = np.expand_dims(rotation, axis=0)
            local_rotation = np.expand_dims(local_rotation, axis=0)
        elif len(raw_size) > 3:
            rotation = rotation.reshape((-1,) + raw_size[-2:])
            local_rotation = local_rotation.reshape((-1,) + raw_size[-2:])

        local_rotation[:, 0] = rotation[:, 0]
        for i in range(1, len(self.parent)):
            local_rotation[:, i] = (R.from_quat(rotation[:, self.parent[i]]).inv() * R.from_quat(rotation[:, i])).as_quat()
        return local_rotation.reshape(raw_size)

    """
    :param rotation - Joint local rotation
    """    
    def rotation_local2global(self, rotation):
        assert rotation.shape[-2] == len(self.joints)
        raw_size = rotation.shape
        global_rotation = np.zeros_like(rotation)
        if len(raw_size) == 2:
            rotation = np.expand_dims(rotation, axis=0)
            global_rotation = np.expand_dims(global_rotation, axis=0)
        elif len(raw_size) > 3:
            rotation = rotation.reshape((-1,) + raw_size[-2:])
            global_rotation = global_rotation.reshape((-1,) + raw_size[-2:])

        global_rotation[:, 0] = rotation[:, 0]
        for i in range(1, len(self.parent)):
            global_rotation[:, i] = (R.from_quat(global_rotation[:, self.parent[i]]) * R.from_quat(rotation[:, i])).as_quat()
        return global_rotation.reshape(raw_size)

    """
    Forward kinematics
    :param rotation - Joint rotation
    :param hip_position - Hip position
    :param rot_type - Rotation type ["global", "local"]
    """
    def forward_kinematics(self, rotation, hip_position, rot_type="global"):
        assert rot_type in ["global", "local"]
        assert rotation.shape[-2] == len(self.joints)
        assert rotation.shape[:-2] == hip_position.shape[:-1]
        raw_size = rotation.shape
        position = np.zeros(raw_size[:-1] + (3,))
        if len(raw_size) == 2:
            rotation = np.expand_dims(rotation, axis=0)
            position = np.expand_dims(position, axis=0)
        elif len(raw_size) > 3:
            rotation = rotation.reshape((-1,) + rotation.shape[-2:])
            position = position.reshape((-1,) + position.shape[-2:])
            hip_position = hip_position.reshape((-1, 3))
        
        position[:, 0] = hip_position
        
        if rot_type == "local":
            rotation = self.rotation_local2global(rotation)

        for i in range(1, len(self.joints)):
            position[:, i] = R.from_quat(rotation[:, self.parent[i]]).apply(self.skel_offset[i]) + position[:, self.parent[i]]
        position = position.reshape(raw_size[:-1] + (3,))
        return position

    """
    Extract root position
    :param position - Joint global positions
    """
    def comp_rootpos(self, position):
        assert position.shape[-2] == len(self.joints)
        l_sdr, r_sdr = self.joints.index("L_Collar"), self.joints.index("R_Collar")
        hip = 0
        l_leg, r_leg = self.joints.index("L_Hip"), self.joints.index("R_Hip")

        sdr_center = (position[..., l_sdr, :] + position[..., r_sdr, :]) / 2
        leg_center = (position[..., l_leg, :] + position[..., r_leg, :]) / 2
        
        root_pos = (position[..., hip, :] + sdr_center + leg_center) / 3
        root_pos[..., "xyz".index("z")] = 0.
        return root_pos

    """
    Extract character forward 
    :param position - Joint global positions
    """
    def comp_forward(self, position):
        assert position.shape[-2] == len(self.joints)
        l_sdr, r_sdr = self.joints.index("L_Collar"), self.joints.index("R_Collar")
        
        across  = position[..., l_sdr, :] - position[..., r_sdr, :]
        forward = np.cross(across, Pivots.ZAxis)
        return vector_norm(forward)

    """
    Transform rotvec to quaternion
    :param rotvec - Joint rotation 
    """
    def rotvec2quat(self, rotvec):
        raw_size = rotvec.shape
        frame = raw_size[0]
        assert len(raw_size) in (2, 3)
        if len(raw_size) == 2:
            assert raw_size[-1] % 3 == 0 and raw_size[-1] > 3
        elif len(raw_size) == 3:
            assert raw_size[-1] == 3
        rotvec = rotvec.reshape(frame, -1, 3)[:, :len(self.joints)].reshape(-1, 3)
        quat = R.from_rotvec(rotvec).as_quat().reshape(frame, -1, 4)
        return quat

    """
    Forward transform rotation
    :param rotation - Joint rotation represented as Quaternion
    :param transform - Rotation Transform, type as utils.quaternions.Quaternions
    :param rot_type - Rotation type ["global", "local"]
    """
    def rotation_forward_transform(self, rotation, transform, rot_type="global"):
        assert rot_type in ["global", "local"]
        assert rotation.shape[-2] == len(self.joints) and len(rotation.shape) in (2, 3)
        if rot_type == "global":
            rotation = Quaternions(rotation)
            return (transform * rotation).qs
        elif rot_type == "local":
            new_rotation = np.zeros_like(rotation)
            new_rotation[..., 1:, :] = rotation[..., 1:, :]
            hip_rotation = Quaternions(rotation[..., 0, :])
            new_rotation[..., 0, :] = (transform * hip_rotation).qs
            return new_rotation

    """
    Forward transform position
    :param position - Joint position
    :param transform - Rotation Transform, type as utils.quaternions.Quaternions
    :param pos_type - Position type ["global", "local"]
    """
    def position_forward_transform(self, position, transform, pos_type="global"):
        assert pos_type in ["global", "local"]
        assert position.shape[-2] == len(self.joints) and len(position.shape) in (2, 3)

        if pos_type == "global":
            return transform.rot(position)
        elif pos_type == "local":
            new_position = np.zeros_like(position)
            new_position[..., 1:, :] = position[..., 1:, :]
            new_position[..., 0, :] = transform.rot(position[..., 0, :])
            return new_position


if __name__ == "__main__":

    """
    test rotation local2global
    """
#     joint_info = json.load(open('/home/data/Motion3D/Human3.6M_SMPL/joint_info.json'))
#     sct = SkeletonCoordinatesTransform(joint_info["joints"], joint_info["parent"], joint_info["skel_offset"])
#     data = json.load(open('/home/data/Motion3D/AMASS/walking_poses.json'))
#     rotation = np.stack([data[j] for j in joint_info["joints"]], axis=1)
#     global_rotation = sct.rotation_local2global(rotation)
    
#     global_data_json = {}
#     for i, j in enumerate(joint_info["joints"]):
#         global_data_json[j] = global_rotation[:, i].tolist()
#     json.dump(global_data_json, open('/home/data/Motion3D/AMASS/walking_poses_global.json', 'w'))
    """
    test forward kinematics
    """
    # joint_info = json.load(open('/home/data/Motion3D/AMASS/joint_info.json'))
    # sct = SkeletonCoordinatesTransform(joint_info["joints"], joint_info["parent"], joint_info["skel_offset"])
    
    # data = np.load("/home/data/Motion3D/AMASS/CMU/01/01_07_poses.npz")
    # frame = len(data["poses"])
    # rotation = data["poses"].reshape(frame, -1, 3)[:, :24].reshape(-1, 3)
    # rotation = R.from_rotvec(rotation).as_quat().reshape(-1, 24, 4)
    # skel_global_position = sct.forward_kinematics(rotation, data["trans"], rot_type="local")
    
    # lsdr, rsdr = sct.joints.index("L_Collar"), sct.joints.index("R_Collar")
    # across = skel_global_position[0, lsdr] - skel_global_position[0, rsdr]
    # forward = np.cross(across, np.array([0, 0, 1.]))
    # forward = vector_norm(forward)
    # print("Init forward:", forward)

    """
    test rotation forward transform
    """
    # joint_info = json.load(open('/home/data/Motion3D/AMASS/joint_info.json'))
    # sct = SkeletonCoordinatesTransform(joint_info["joints"], joint_info["parent"], joint_info["skel_offset"])

    # data = np.load("/home/data/Motion3D/AMASS_60FPS/CMU/01/01_07_poses.npz")
    # local_rotation = sct.rotvec2quat(data['poses'])

    # transform = R.from_euler("xyz", [90, 45, 45], degrees=True)
    # new_local_rotation = sct.rotation_forward_transform(local_rotation, Quaternions(transform.as_quat()), rot_type="local")
    # assert np.all(local_rotation[:, 1:] == new_local_rotation[:, 1:])
    # print("New local hip rotation[:4]:", new_local_rotation[:4, 0])
    # print("New local hip rotation[:4] from scipy:", (transform * R.from_quat(local_rotation[:4, 0])).as_quat())
    
    """
    test position forward transform
    """
    joint_info = json.load(open('/home/data/Motion3D/AMASS/joint_info.json'))
    sct = SkeletonCoordinatesTransform(joint_info["joints"], joint_info["parent"], joint_info["skel_offset"])

    data = np.load("/home/data/Motion3D/AMASS_60FPS/CMU/01/01_07_poses.npz")
    local_rotation = sct.rotvec2quat(data['poses'])
    global_pos = sct.forward_kinematics(local_rotation, data['trans'], rot_type="local")

    transform = R.from_euler("xyz", [90, 45, 45], degrees=True)
    new_global_pos = sct.position_forward_transform(global_pos, Quaternions(transform.as_quat())[np.newaxis], pos_type="global")
    print("New local hip position[:4]:", new_global_pos[:4, 0])
    print("New local hip position[:4] from scipy:", transform.apply(global_pos[:4, 0]))

    print("New local l_hip position[:4]:", new_global_pos[:4, 1])
    print("New local l_hip position[:4] from scipy:", transform.apply(global_pos[:4, 1]))