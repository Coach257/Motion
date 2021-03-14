import argparse
import os
import time

from addict import Dict
import torch
from torch.utils.tensorboard import SummaryWriter
import yaml

from datasets.data_loader import build_dataloader
from model.tg_model import TGModel
from utils.logger import Logger
from model.losses import Loss


def parse_args():
    parser = argparse.ArgumentParser(description="Train motion in-between model")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--resume_from", help="the checkpoint file to resume from")
    return parser.parse_args()


class Trainer(object):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.logger = Logger("trainer")
        cfg.loss.device = torch.device(cfg.device)
        self.loss_func = Loss(cfg.loss)
        os.makedirs(cfg.train.save_path, exist_ok=True)
        os.makedirs(cfg.train.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=cfg.train.log_dir)

    def train(self):
        cfg = self.cfg
        device = torch.device(cfg.device)
        
        data_loader = build_dataloader(cfg.data.data_path, cfg.data.joint_file,
            cfg.train.batch_size, True, cfg.train.min_trans, cfg.train.max_trans)
        dataset = data_loader.dataset

        state_dim, offset_dim, target_dim, out_dim = dataset.get_dim()
        
        cfg.model.state_dim = state_dim
        cfg.model.offset_dim = offset_dim
        cfg.model.target_dim = target_dim
        cfg.model.out_dim = out_dim
        cfg.model.max_trans = cfg.train.max_trans
        
        model = TGModel(cfg.model)
        model.to(device=device)

        joint_num = len(dataset.joint2id)

        if cfg.train.optimizer.type == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.optimizer.lr,
                betas=(cfg.train.optimizer.get("beta1", 0.9), cfg.train.optimizer.get("beta2", 0.999)), 
                weight_decay=cfg.train.optimizer.get("weight_decay", 0),
                amsgrad=cfg.train.optimizer.get("amsgrad", False))
        elif cfg.train.optimizer.type == "AdamW":
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.optimizer.lr,
                betas=(cfg.train.optimizer.get("beta1", 0.9), cfg.train.optimizer.get("beta2", 0.999)), 
                weight_decay=cfg.train.optimizer.get("weight_decay", 0),
                amsgrad=cfg.train.optimizer.get("amsgrad", False))
        else:
            self.logger.logging("The optimizer type {} is not supported".format(cfg.train.optimizer.type))

        losses = 0.
        step = 0
        metric = 0.
        best_metric = 1e7
        start_epoch = 0
        if cfg.resume_from is not None:
            state = torch.load(cfg.resume_from)
            model.load_state_dict(state['net'])
            optimizer.load_state_dict(state['optimizer'])
            start_epoch = state['epoch'] + 1
            best_metric = state['best_metric']
            step = state['step']

        for e in range(start_epoch, cfg.train.epoch):
            model.train()
            self.logger.logging("Start Epoch %d" % e)
            metric = 0.
            for input, target, mask in data_loader:
                for k in input.keys():
                    input[k] = input[k].to(device)
                state_input = torch.cat((input["contact"],
                    input["root_vel"],
                    input["rotation"],
                    input["joint_pos_vel"],
                    input["joint_pos_acc"],
                    input["joint_rotation_vel"],
                    input["joint_rotation_acc"]), dim=-1)
                offset_input = torch.cat((input["root_to_target"],
                    input["rotation_to_target"]), dim=-1)
                target_input = input["target_rotation"]
                mask = mask.to(device)
                input_len = mask.sum(dim=-1).max(dim=-1)[0]
                pred = model(state_input, offset_input, target_input, mask, input_len)

                pred_contact = pred[..., :4]
                pred_root = pred[..., 4:7]
                pred_rot = pred[...,7:7+joint_num*4]
                pred_joint_pos_vel = pred[...,joint_num*4+7:joint_num*7+7]
                pred_joint_pos_acc = pred[...,joint_num*7+7:joint_num*10+7]
                pred_joint_rotation_vel = pred[...,joint_num*10+7:joint_num*14+7]
                pred_joint_rotation_acc = pred[...,joint_num*14+7:joint_num*18+7]
                # pred_rot = pred[..., :joint_num*4]
                # pred_root = pred[..., joint_num*4:joint_num*4+3]
                # pred_contact = pred[..., -(out_dim - (joint_num*4+3)):]

                if cfg.loss.rot_diff:
                    pred_info = {"rotation": pred_rot + input["rotation"],
                        "root_vel": pred_root,
                        "contact": pred_contact,
                        "joint_pos_vel":pred_joint_pos_vel,
                        "joint_pos_acc":pred_joint_pos_acc,
                        "joint_rotation_vel":pred_joint_rotation_vel,
                        "joint_rotation_acc":pred_joint_rotation_acc
                        }
                else:
                    pred_info = {"rotation": pred_rot,
                        "root_vel": pred_root,
                        "contact": pred_contact,
                        "joint_pos_vel":pred_joint_pos_vel,
                        "joint_pos_acc":pred_joint_pos_acc,
                        "joint_rotation_vel":pred_joint_rotation_vel,
                        "joint_rotation_acc":pred_joint_rotation_acc
                        }
                lengths = mask.sum(dim=-1).max(dim=-1)[0]
                mask = torch.arange(state_input.size(1), device=device) < lengths.unsqueeze(-1)
                loss = self.loss_func.compute_loss(pred_info, target, mask)
                self.writer.add_scalar("loss", loss.item(), step)

                losses += loss.item()
                step += 1
                metric += loss.item()
                if step % cfg.train.disp_every == 0:
                    self.logger.logging("Step {}, loss is {}".format(step, losses / cfg.train.disp_every))
                    losses = 0.
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            metric /= len(data_loader)
            self.writer.add_scalar("metric", metric, e)
            self.logger.logging("Epoch {}, the metric is {}".format(e, metric))
            if metric < best_metric:
                self.logger.logging("Epoch {}, the best metric {} --> {}".format(e, best_metric, metric))
                best_metric = metric
                state = {}
                state["config"] = cfg
                state["net"] = model.state_dict()
                torch.save(state, os.path.join(cfg.train.save_path, "best.m"))
            
            if e >= cfg.train.epoch - 10:
                state = {}
                state["config"] = cfg
                state["net"] = model.state_dict()
                torch.save(state, os.path.join(cfg.train.save_path, "model_{}.m".format(e))) 
            
            state = {}
            state['config'] = cfg
            state['net'] = model.state_dict()
            state['optimizer'] = optimizer.state_dict()
            state['best_metric'] = best_metric
            state['epoch'] = e
            state['step'] = step
            torch.save(state, os.path.join(cfg.train.save_path, "checkpoint.m"))


def main():
    args = parse_args()
    cfg = Dict(yaml.safe_load(open(args.config)))
    cfg.device = args.gpu_id
    cfg.resume_from = args.resume_from
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()




