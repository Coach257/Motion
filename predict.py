import argparse
import json
import os
import pickle as pk

from addict import Dict
import numpy as np
import torch
import yaml

from datasets.data_loader import subsequent_mask
from model.tg_model import TGModel
from utils.coordinates_transform import SkeletonCoordinatesTransform
from utils.pivots import Pivots
from utils.quaternions import Quaternions


def parse_args():
    parser = argparse.ArgumentParser(description="Predict motion with in-between model")
    parser.add_argument("config", help="Prediction config file path")
    parser.add_argument("--result_path", help="The path for saving result motion")
    parser.add_argument("--gpu_id", type=int, default=0)
    return parser.parse_args()


class Predictor(object):
    """
    :param model_path - The path of trained model
    """
    def __init__(self, cfg):
        self.cfg = cfg
        state_dict = torch.load(self.cfg.model_path)
        self.model_cfg = state_dict['config']
        
        self.model = TGModel(self.model_cfg.model)
        self.model.load_state_dict(state_dict['net'])
        self.model.eval()
        self.device = torch.device(self.cfg.device)
        self.model.to(self.device)
        self.joint_info = json.load(open(self.cfg.joint_file))
        self.sct = SkeletonCoordinatesTransform(self.joint_info['joints'],
            self.joint_info['parent'],
            self.joint_info['skel_offset'])
    
    @torch.no_grad()
    def predict(self, start_trans, start_poses, tgt_trans, tgt_poses, start_foot_contact, trans_length, add_tgt_noise):
        def reshape2D(tensor):
            return tensor.reshape(tensor.shape[0], -1)
        
        # start_local_rotation = self.sct.rotvec2quat(start_poses)
        start_local_rotation = start_poses
        # tgt_local_rotation = self.sct.rotvec2quat(tgt_poses)
        tgt_local_rotation = tgt_poses
        
        start_global_pos = self.sct.forward_kinematics(start_local_rotation, start_trans, rot_type="local")
        tgt_global_pos = self.sct.forward_kinematics(tgt_local_rotation, tgt_trans, rot_type='local')

        start_forward = self.sct.comp_forward(start_global_pos[0])
        # print("Forward shape:", start_forward.shape)
        glo2forward = Quaternions.between(start_forward, Pivots.ZAxis)
        
        init_root_pos = start_global_pos[:, 0]
        start_local_rotation = self.sct.rotation_forward_transform(start_local_rotation, glo2forward, rot_type="local")
        start_global_pos = self.sct.position_forward_transform(start_global_pos - init_root_pos[0], glo2forward[np.newaxis], pos_type="global")
        tgt_local_rotation = self.sct.rotation_forward_transform(tgt_local_rotation, glo2forward, rot_type="local")
        tgt_global_pos = self.sct.position_forward_transform(tgt_global_pos - init_root_pos[0], glo2forward[np.newaxis], pos_type="global")
        start_root_pos = start_global_pos[:, 0]
        tgt_root_pos = tgt_global_pos[:, 0]
        assert tgt_root_pos.shape[0] == 1

        input_root_pos = torch.Tensor(start_root_pos).to(self.device)
        last_root_pos = input_root_pos[-1]
        tgt_root_pos = torch.Tensor(tgt_root_pos).to(self.device)
        input_local_rotation = torch.Tensor(start_local_rotation).to(self.device)
        last_local_rotation = input_local_rotation[-1]
        tgt_local_rotation = torch.Tensor(tgt_local_rotation).to(self.device)

        input_root_vel = torch.cat((torch.zeros((1, 3), device=self.device),
            input_root_pos[1:] - input_root_pos[:-1]), dim=0)
        input_foot_contact = torch.Tensor(start_foot_contact).to(self.device)

        root_to_target = input_root_pos - tgt_root_pos
        rotation_to_target = input_local_rotation - tgt_local_rotation

        start_global_pos = torch.Tensor(start_global_pos).to(self.device)

        joint_pos_vel = torch.cat((torch.zeros((1,24,3),device=self.device),start_global_pos[1:]-start_global_pos[:-1]),dim=0)

        joint_rotation_vel = torch.cat((torch.zeros((1,24,4),device=self.device),input_local_rotation[1:]-input_local_rotation[:-1]),dim=0)

        joint_pos_acc = torch.cat((torch.zeros((1,24,3),device=self.device),joint_pos_vel[1:]-joint_pos_vel[:-1]),dim=0)
        joint_rotation_acc = torch.cat((torch.zeros((1,24,4),device=self.device),joint_rotation_vel[1:]-joint_rotation_vel[:-1]),dim=0)

        input_len = torch.tensor([trans_length + len(input_local_rotation) - 1], device=self.device)
        joint_num = len(self.joint_info['joints'])
        for step in range(0, trans_length - 1):
            input_state = torch.cat((
                reshape2D(input_foot_contact),
                reshape2D(input_root_vel),
                reshape2D(input_local_rotation),
                reshape2D(joint_pos_vel),
                reshape2D(joint_pos_acc),
                reshape2D(joint_rotation_vel),
                reshape2D(joint_rotation_acc)
            ), dim=-1)
            
            input_offset = torch.cat((
                root_to_target,
                reshape2D(rotation_to_target)
            ), dim=-1)

            input_target = reshape2D(tgt_local_rotation).repeat(input_state.size(0), 1)
            mask = subsequent_mask(input_state.size(0)).to(self.device)

            output = self.model(input_state.unsqueeze(0),
                input_offset.unsqueeze(0),
                input_target.unsqueeze(0),
                mask,
                input_len,
                add_tgt_noise=add_tgt_noise)[0, -1]

            pred_contact = (output[:4]>0).type(torch.float32)
            pred_root_vel = output[4:7]
            pred_rot = output[7:7+joint_num*4].reshape(joint_num, 4)

            pred_joint_pos_vel = output[joint_num*4+7:joint_num*7+7].reshape(joint_num,3)
            pred_joint_pos_acc = output[joint_num*7+7:joint_num*10+7].reshape(joint_num,3)
            pred_joint_rotation_vel = output[joint_num*10+7:joint_num*14+7].reshape(joint_num,4)
            pred_joint_rotation_acc = output[joint_num*14+7:joint_num*18+7].reshape(joint_num,4)

            input_root_vel = torch.cat((input_root_vel,
                pred_root_vel.unsqueeze(0)), dim=0)
            input_foot_contact = torch.cat((input_foot_contact,
                pred_contact.unsqueeze(0)), dim=0)

            joint_pos_vel = torch.cat((joint_pos_vel,pred_joint_pos_vel.unsqueeze(0)),dim=0)
            joint_pos_acc = torch.cat((joint_pos_acc,pred_joint_pos_acc.unsqueeze(0)),dim=0)
            joint_rotation_vel = torch.cat((joint_rotation_vel,pred_joint_rotation_vel.unsqueeze(0)),dim=0)
            joint_rotation_acc = torch.cat((joint_rotation_acc,pred_joint_rotation_vel.unsqueeze(0)),dim=0)

            if self.model_cfg.loss.rot_diff:
                print("rot_diff")
                last_local_rotation = last_local_rotation + pred_rot
                
            else:
                last_local_rotation = pred_rot
            last_local_rotation = last_local_rotation / torch.sqrt(torch.sum(last_local_rotation ** 2, dim=-1, keepdim=True))
            input_local_rotation = torch.cat((input_local_rotation,
                last_local_rotation.unsqueeze(0)), dim=0)
            
            last_root_pos = last_root_pos + pred_root_vel
            input_root_pos = torch.cat((input_root_pos,
                last_root_pos.unsqueeze(0)), dim=0)
            root_to_target = torch.cat((root_to_target,
                last_root_pos - tgt_root_pos), dim=0)
            rotation_to_target = torch.cat((rotation_to_target,
                last_local_rotation - tgt_local_rotation), dim=0)

        trans = input_root_pos[-(trans_length - 1):].cpu().numpy()
        print("Trans shape:", trans.shape)
        trans = glo2forward.inv().rot(trans) + init_root_pos[0]
        poses = input_local_rotation[-(trans_length-1):].cpu().numpy()
        
        poses = self.sct.rotation_forward_transform(poses, glo2forward.inv(), rot_type="local")
        
        # return np.concatenate((start_trans, trans, tgt_trans.repeat(60, axis=0)), axis=0),  np.concatenate((start_poses[:, :72], reshape2D(self.sct.quat2rotvec(poses)), tgt_poses[:, :72].repeat(60, axis=0)), axis=0)
        return np.concatenate((start_trans, trans, tgt_trans.repeat(60, axis=0)), axis=0),  np.concatenate((start_poses,poses, tgt_poses.repeat(60, axis=0)), axis=0)


def main():
    args = parse_args()
    cfg = Dict(yaml.safe_load(open(args.config)))
    cfg.device = args.gpu_id
    
    keyframe = json.load(open(cfg.keyframe_path))
    
    start_trans, start_poses = np.array(keyframe['start_trans']), np.array(keyframe['start_poses'])
    tgt_trans, tgt_poses = np.array(keyframe['tgt_trans']), np.array(keyframe['tgt_poses'])
    start_foot_contact = np.array(keyframe['start_foot_contact'])

    predictor = Predictor(cfg)
    pred_trans, pred_poses = predictor.predict(start_trans, start_poses, tgt_trans, tgt_poses, start_foot_contact, trans_length=cfg.trans_length, add_tgt_noise=cfg.add_tgt_noise)
    os.makedirs(os.path.dirname(args.result_path), exist_ok=True)
    pk.dump({"cfg": cfg, "trans": pred_trans, "poses": pred_poses}, open(args.result_path, 'wb'))


if __name__ == "__main__":
    main()
