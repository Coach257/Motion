import math
import numpy as np


def resample(data, orig_fr, tgt_fr):
    feat_shape = data.shape[1:]
    frame = data.shape[0]
    data = data.reshape(frame, -1)
    
    step = orig_fr / tgt_fr
    indices = np.arange(0, frame - 1, step=step)

    left_value = data[np.floor(indices).astype(np.int32)]
    right_value = data[np.ceil(indices).astype(np.int32)]
    prop = (indices % 1)[:, np.newaxis]

    value = left_value * (1 - prop) + right_value * prop
    return value.reshape((-1,) + feat_shape)


def sample_segment(data, length):
    start_index = np.random.randint(0, len(data)-length+1)
    return data[start_index:start_index+length]
    