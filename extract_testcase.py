import json
import pickle as pk
import json
from addict import Dict
import yaml
import numpy as np
import argparse
from utils.coordinates_transform import SkeletonCoordinatesTransform
from utils.foot_contact import FootContact

joint_info = json.load(open('/home/data/Motion3D/motionjson/joint_info.json'))


def parse_args():
    parser = argparse.ArgumentParser(description="Extract testcase")
    parser.add_argument("config", help="Extraction config file path")
    return parser.parse_args()

def extract_testcase(raw_file, seed_frames, target_frame, joint_info):
    joints = joint_info['joints']
    parent = joint_info['parent']
    skel_offset = joint_info['skel_offset']

    sct = SkeletonCoordinatesTransform(joints, parent, skel_offset)
    fc = FootContact(joints, 3e-3)

    data = json.load(open(raw_file))
    keyframe = {}
    keyframe['seed_frames'] = seed_frames
    keyframe['target_frame'] = target_frame
    keyframe['raw_file'] = raw_file
    keyframe['start_trans'] = data['root_positions'][seed_frames[0]:seed_frames[1]]
    keyframe['start_poses'] = data['rotations'][seed_frames[0]:seed_frames[1]]
    keyframe['tgt_trans'] = data['root_positions'][target_frame[0]:target_frame[1]]
    keyframe['tgt_poses'] = data['rotations'][target_frame[0]:target_frame[1]]

    local_rotation = np.array(data['rotations'])
    global_pos = sct.forward_kinematics(local_rotation, np.array(data['root_positions']), rot_type="local")
    foot_contact = fc.judge_by_vel(global_pos)
    keyframe['start_foot_contact'] = foot_contact[seed_frames[0]:seed_frames[1]].tolist()

    return keyframe
    
def main():
    args = parse_args()
    cfg = Dict(yaml.safe_load(open(args.config)))
    data_file = cfg.data_file
    start_n = cfg.start_n
    end_n = cfg.end_n
    target = cfg.target
    joint_info = json.load(open(cfg.joint_file))
    testcase = extract_testcase(data_file,(start_n,end_n),(target-1,target),joint_info)
    json.dump(testcase,open("experiments/example/{}_{}:{}:{}.json".format(data_file.split("/")[-1].replace(".json",""),start_n,end_n,target-1),"w"))

if __name__ == "__main__":
    main()
    