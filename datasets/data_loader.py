import time

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from .motion_dataset import MotionDataset



def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def make_std_mask(tgt, pad):
    "Create a mask to hide padding and future words."
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
    return tgt_mask


def batch_fn(data, lengths):
    st = time.time()
    keys = list(data[0].keys())
    max_length = max(lengths)
    batch = {}
    for k in keys:
        batch_k = []
        for i in range(len(data)):
            batch_k.append(torch.cat((data[i][k], torch.zeros( (max_length-data[i][k].size(0),) + data[i][k].shape[1:] ) ), dim=0))
        batch[k] = torch.stack(batch_k, dim=0)
    # print(time.ctime(), "batch_fn time:", time.time() - st)
    return batch


def collate_fn(batch):
    lengths = []
    input, output, _ = batch[0]
    k = list(input.keys())[0]
    for i in range(len(batch)):
        lengths.append(len(batch[i][0][k]))
    max_length = max(lengths)
    mask = torch.tensor([[1] * l + [0] * (max_length - l) for l in lengths]).type(torch.bool)
    mask = make_std_mask(mask, 0)

    inputs, outputs = [], []
    for i in range(len(batch)):
        inputs.append(batch[i][0])
        outputs.append(batch[i][1])
    
    ## test
    anims = [batch[i][2] for i in range(len(batch))]
    # print("Batch anim indices:", anims)
    ##
    return batch_fn(inputs, lengths), batch_fn(outputs, lengths), mask


def build_dataloader(data_path, joint_file, batch_size, is_training, min_trans, max_trans):
    dataset = MotionDataset(data_path, joint_file, is_training, min_trans, max_trans)
    if is_training:
        sampler = WeightedRandomSampler(dataset.data_weight, len(dataset), True)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                    num_workers=4, collate_fn=collate_fn, drop_last=True, sampler=sampler)
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False,
            num_workers=4, collate_fn=collate_fn, drop_last=False)


if __name__ == "__main__":
    data_loader = build_dataloader("/home/data/Motion3D/motionjson/", "joint_info.json", 5,True, 30, 120)
    input,target,mask = next(iter(data_loader))
    import ipdb;ipdb.set_trace()