import torch
from torch import nn


class Loss(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.device
        self.weight = self.cfg.weight

    def compute_loss(self, pred, target, mask):
        rotation = pred["rotation"]
        root = pred["root_vel"]
        contact = torch.sigmoid(pred["contact"])
        joint_pos_vel = pred["joint_pos_vel"]
        joint_pos_acc = pred["joint_pos_acc"]
        joint_rotation_vel = pred["joint_rotation_vel"]
        joint_rotation_acc = pred["joint_rotation_acc"]

        target_rotation = target["rotation"].to(self.device)
        target_root = target["root_vel"].to(self.device)
        target_contact = target["contact"].to(self.device)
        target_joint_pos_vel = target["joint_pos_vel"].to(self.device)
        target_joint_pos_acc = target["joint_pos_acc"].to(self.device)
        target_joint_rotation_vel = target["joint_rotation_vel"].to(self.device)
        target_joint_rotation_acc = target["joint_rotation_acc"].to(self.device)

        if self.cfg.type == "l2":
            return self.weight.local_rot * l2_loss(rotation, target_rotation, mask=mask) + self.weight.root * l2_loss(root, target_root, mask=mask) + self.weight.contact * l2_loss(contact, target_contact, mask=mask)+self.weight.joint_pos_vel * l2_loss(joint_pos_vel,target_joint_pos_vel,mask=mask) + self.weight.joint_pos_acc * l2_loss(joint_pos_acc, target_joint_pos_acc, mask=mask) + self.weight.joint_rotation_vel * l2_loss(joint_rotation_vel,target_joint_rotation_vel,mask=mask)+ self.weight.joint_rotation_acc * l2_loss(joint_rotation_acc,target_joint_rotation_acc,mask=mask)
        elif self.cfg.type == "l1":
            return self.weight.local_rot * l1_loss(rotation, target_rotation, mask=mask) + self.weight.root * l1_loss(root, target_root, mask=mask) + self.weight.contact * l1_loss(contact, target_contact, mask=mask)+self.weight.joint_pos_vel * l1_loss(joint_pos_vel,target_joint_pos_vel,mask=mask) + self.weight.joint_pos_acc * l1_loss(joint_pos_acc, target_joint_pos_acc, mask=mask) + self.weight.joint_rotation_vel * l1_loss(joint_rotation_vel,target_joint_rotation_vel,mask=mask)+ self.weight.joint_rotation_acc * l1_loss(joint_rotation_acc,target_joint_rotation_acc,mask=mask)


def l2_loss(pred, target, mask=None):
    loss = torch.pow(pred - target, 2)
    if mask is not None:
        loss = loss * mask.unsqueeze(-1)
    return loss.sum() / (mask.sum() * pred.size(-1))


def l1_loss(pred, target, mask=None):
    loss = torch.abs(pred - target)
    if mask is not None:
        loss = loss * mask.unsqueeze(-1)
    return loss.sum() / (mask.sum() * pred.size(-1))