import glob
import json
import os
import pickle as pk
import time
import datetime

import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
from torch.utils.data import Dataset

from utils.coordinates_transform import SkeletonCoordinatesTransform
from utils.data_util import segment_filter, get_all_files, get_all_lengths, filter_files
from utils.foot_contact import FootContact
from utils.quaternions import Quaternions
from utils.pivots import Pivots
from utils.sample_util import sample_segment
from utils.cache_schedule import CacheSchedule


class MotionDataset(Dataset):
    def __init__(self, data_path, joint_file, is_training, min_trans, max_trans):
        self.data_path = data_path
        self.data_files = get_all_files(data_path, "json")
        print("The number of data files is %d" % len(self.data_files))
    
        self.is_training = is_training
        self.min_trans = min_trans
        self.max_trans = max_trans
        joint_info = json.load(open(joint_file))
        self.joints = joint_info['joints']
        self.parent = joint_info['parent']
        self.skel_offset = joint_info['skel_offset']
        self.sct = SkeletonCoordinatesTransform(self.joints, self.parent, self.skel_offset)
        self.fc = FootContact(self.joints, 3e-3)
        self.joint2id = {}
        for id, joint in enumerate(self.joints):
            self.joint2id[joint] = id
        self.cache_scheduler = CacheSchedule(0, data_path, self.data_files, "json")

        if not is_training:
            assert min_trans == max_trans

        self.data_length = get_all_lengths(data_path, self.data_files,"json")
        print("Total frames in training dataset is %d" % sum(self.data_length))
        s = sum(self.data_length)
        self.data_weight = [d / s for d in self.data_length]
    def __getitem__(self, index):
        if self.is_training:
            st = time.time()
            ## while True:
            # random select a animation 
            # anim_index = np.random.choice(np.arange(0, len(self.data_files)), p=self.data_weight)
            # print("The animation file:", self.data_files[anim_index])
            # random select trans length
            # trans_length = np.random.randint(self.min_trans, self.max_trans + 1)
            
            # print("trans_length:", trans_length)
            # load anim
            data = self.cache_scheduler.load(os.path.join(self.data_path, self.data_files[index]), index)
            n = len(data['rotations'])
            if n <= self.min_trans:
                trans_length = n-1
            else :
                trans_length = torch.randint(self.min_trans,min(n-1,self.max_trans)+ 1,(1,))[0]
            # random select start_index
            # while len(data['rotations']) - trans_length <= 0:  
            start_index = torch.randint(0, len(data['rotations']) - trans_length, (1,)).item()
            # print("start_index:", start_index)
            trans, poses = data['root_positions'][start_index:start_index+trans_length+1], data['rotations'][start_index:start_index+trans_length+1]
            trans = np.array(trans)
            poses = np.array(poses)
            # local_rotations = self.sct.rotvec2quat(poses)
            local_rotations = poses
            import ipdb;ipdb.set_trace()
            # st = time.time()
            global_pos = self.sct.forward_kinematics(local_rotations, trans, rot_type="local")
            # print("Forward kinematics time:", time.time() - st)
            
            # if segment_filter(global_pos, 0.1):
            #     # print("Pass")
            #     break
            # else:
            #     pass
            # print(time.ctime(), "Loop find data:", time.time() - st)
        else:
            i = 0
            start_index = index
            while start_index >= self.data_length[i] - self.min_trans:
                start_index -= (self.data_length[i] - self.min_trans)
                i += 1
            anim_index = i
            assert self.min_trans == self.max_trans
            trans_length = self.min_trans
            data = self.cache_scheduler.load(os.path.join(self.data_path, self.data_files[i]))
            trans, poses = data['trans'][start_index:start_index+trans_length+1], data['poses'][start_index:start_index+trans_length+1]
            local_rotations = self.sct.rotvec2quat(poses)
            global_pos = self.sct.forward_kinematics(local_rotations, trans, rot_type="local")

        root_pos = self.sct.comp_rootpos(global_pos)

        start_forward = self.sct.comp_forward(global_pos[0])
        # print("Forward shape:", start_forward.shape)
        glo2forward = Quaternions.between(start_forward, Pivots.ZAxis)
        
        # st = time.time()
        local_rotations = self.sct.rotation_forward_transform(local_rotations, glo2forward, rot_type="local")
        global_pos = self.sct.position_forward_transform(global_pos - root_pos[0], glo2forward[np.newaxis], pos_type="global")
        # print("Forward transform time:", time.time() - st)
        
        input_local_rotations = local_rotations[:-1]
        output_local_rotations = local_rotations[1:]

        root_pos = self.sct.comp_rootpos(global_pos)

        input_root_vel = root_pos[1:trans_length] - root_pos[0:trans_length-1]
        input_root_vel = np.concatenate((np.zeros((1, 3)), input_root_vel), axis=0)
        output_root_vel = root_pos[1:trans_length+1] - root_pos[0:trans_length]

        foot_contact = self.fc.judge_by_vel(global_pos)
        input_contact = foot_contact[0:-1]
        output_contact = foot_contact[1:]

        local_rotations_to_target = local_rotations[:-1] - local_rotations[-1]
        root_pos_to_target = root_pos[:-1] - root_pos[-1]

        joint_pos_vel = global_pos[1:trans_length+1] - global_pos[0:trans_length]
        joint_pos_vel = np.concatenate((np.zeros((1,24,3)),joint_pos_vel),axis=0)
        input_joint_pos_vel = joint_pos_vel[0:trans_length]
        output_joint_pos_vel = joint_pos_vel[1:trans_length+1]
        input_joint_pos_acc = joint_pos_vel[1:trans_length] - joint_pos_vel[0:trans_length-1]
        input_joint_pos_acc = np.concatenate((np.zeros((1,24,3)),input_joint_pos_acc),axis=0)
        output_joint_pos_acc = joint_pos_vel[1:trans_length+1] - joint_pos_vel[0:trans_length]

        joint_rotation_vel = local_rotations[1:trans_length+1] - local_rotations[0:trans_length]
        joint_rotation_vel = np.concatenate((np.zeros((1,24,4)),joint_rotation_vel),axis=0)
        input_joint_rotation_vel = joint_rotation_vel[0:trans_length]
        output_joint_rotation_vel = joint_rotation_vel[1:trans_length+1]
        input_joint_rotation_acc = joint_rotation_vel[1:trans_length] - joint_rotation_vel[0:trans_length-1]
        input_joint_rotation_acc = np.concatenate((np.zeros((1,24,4)),input_joint_rotation_acc),axis=0)
        output_joint_rotation_acc = joint_rotation_vel[1:trans_length+1] - joint_rotation_vel[0:trans_length]

        def reshape2D(tensor):
            return tensor.reshape(tensor.shape[0], -1)

        return {
            "contact": torch.Tensor(reshape2D(input_contact)),
            "root_vel": torch.Tensor(reshape2D(input_root_vel)),
            "rotation": torch.Tensor(reshape2D(input_local_rotations)),
            "joint_pos_vel": torch.Tensor(reshape2D(input_joint_pos_vel)),
            "joint_pos_acc": torch.Tensor(reshape2D(input_joint_pos_acc)),
            "joint_rotation_vel":torch.Tensor(reshape2D(input_joint_rotation_vel)),
            "joint_rotation_acc":torch.Tensor(reshape2D(input_joint_rotation_acc)),
            "rotation_to_target": torch.Tensor(reshape2D(local_rotations_to_target)),
            "root_to_target": torch.Tensor(reshape2D(root_pos_to_target)),
            "target_rotation": torch.Tensor(reshape2D(local_rotations[-1:].repeat(input_contact.shape[0], axis=0)))
        }, {
            "contact": torch.Tensor(reshape2D(output_contact)),
            "root_vel": torch.Tensor(reshape2D(output_root_vel)),
            "rotation": torch.Tensor(reshape2D(output_local_rotations)),
            "joint_pos_vel": torch.Tensor(reshape2D(output_joint_pos_vel)),
            "joint_pos_acc": torch.Tensor(reshape2D(output_joint_pos_acc)),
            "joint_rotation_vel": torch.Tensor(reshape2D(output_joint_rotation_vel)),
            "joint_rotation_acc": torch.Tensor(reshape2D(output_joint_rotation_acc)),
            "position": torch.Tensor(reshape2D(global_pos[1:]))
        }, start_index

    def __len__(self):
        if self.is_training:
            return len(self.data_files)
        else:
            length = 0
            for i in range(len(self.data_files)):
                length += (self.data_length[i] - self.min_trans)
            return length

    def get_dim(self):
        def feature_dim(tensor):
            return int(np.product(tensor.shape[1:]))
        contact_dim = 4
        rot_dim = 4 * len(self.joints)
        root_dim = 3
        vel_dim = 3 * len(self.joints) + 4 * len(self.joints)
        acc_dim = vel_dim
        state_dim = contact_dim + rot_dim + root_dim + vel_dim + acc_dim
        offset_dim = root_dim + rot_dim
        target_dim = rot_dim
        out_dim = rot_dim + root_dim + contact_dim + vel_dim + acc_dim
        return state_dim, offset_dim, target_dim, out_dim


if __name__ == "__main__":
    dataset = MotionDataset("sample\\data\\", "sample\\joint_info.json", True, 60, 120)
    print("data feature dim:", dataset.get_dim())
    input, target,mask = dataset[0]
    import ipdb;ipdb.set_trace()
    pk.dump(input, open('sample\\data.pkl', 'wb'))