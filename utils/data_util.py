import os
import pickle as pk
import re
import numpy as np
import json
from utils.vector_util import vector_mod
from pathlib import Path


"""
Filter the files
"""
def filter_files(origin_data,file_name):
    filter_file = []
    with open(Path(file_name)) as f:
        for line in f:
            file = re.sub("/home/data/Motion3D/AMASS_60FPS/","",line.strip())
            filter_file.append(file)
    return list(set(origin_data)-set(filter_file))


"""
Filter the segment with little variation
"""
def segment_filter(global_pos, threshold):
    assert len(global_pos.shape) == 3

    variation = vector_mod(global_pos[-1, 0] - global_pos[0, 0]) + np.sum(vector_mod((global_pos[-1] - global_pos[-1, 0]) - (global_pos[0] - global_pos[0, 0])))
    variation /= global_pos.shape[1]

    return variation > threshold


"""
Get all data files
"""
def get_all_files(data_path, ext=".npz"):
    if("json" in ext):
        data_path += ("data\\")
    data_files = []
    if os.path.exists(os.path.join(data_path, "data_files.meta")):
        with open(os.path.join(data_path, "data_files.meta")) as f:
            for line in f:
                data_files.append(line.strip())
        return data_files
    
    for file_dir, _, files in os.walk(data_path):
        for f in files:
            if ext not in f:
                continue
            # print(os.path.join(file_dir, f))
            data_files.append(os.path.join(file_dir, f).replace(data_path, ""))

    with open(os.path.join(data_path, "data_files.meta"), 'w') as f:
        for data_file in data_files:
            f.write(data_file + "\n")

    return data_files


"""
Get all data length
"""
def get_all_lengths(data_path, files,type = "pkl"):

    data_lengths = []
    # if os.path.exists(os.path.join(data_path, "data_lengths.meta")):
    #     with open(os.path.join(data_path, "data_lengths.meta")) as f:
    #         for line in f:
    #             data_lengths.append(int(line.strip()))
    #     return data_lengths
    if(type == "pkl"):
        for f in files:
            data = pk.load(open(os.path.join(data_path, f), "rb"))
            data_lengths.append(len(data['trans']))
    else :
        data_path+="data\\"
        for f in files:
            data = json.load(open(os.path.join(data_path,f)))
            data_lengths.append(len(data['rotations']))
    with open(os.path.join(data_path, "data_lengths.meta"), 'w') as f:
        for length in data_lengths:
            f.write(str(length) + "\n")
    return data_lengths


if __name__ == "__main__":
    data_files = get_all_files("/home/data/Motion3D/AMASS_60FPS/")
    lengths = get_all_lengths("/home/data/Motion3D/AMASS_60FPS/", data_files)
    with open("/home/data/Motion3D/AMASS_60FPS/data_lengths.meta", "w") as f:
        for l in lengths:
            f.write(str(l) + "\n")