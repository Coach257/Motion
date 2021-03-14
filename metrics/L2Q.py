import numpy as np


"""
:param pred_seqs - np.array, shape: (batch, seq, feat_dim), predicted joint rotation/position
:param gt_seqs - np.array, shape: (batch, seq, feat_dim), target joint rotation/position
"""
def compute_l2(pred_seqs, gt_seqs):
    assert pred_seqs.shape == gt_seqs.shape
    return np.linalg.norm(pred_seqs - gt_seqs, ord=2, axis=2).mean()