import numpy as np


def compute_npss(pred_seqs, gt_seqs):
    ''' 
    input
        pred/gt (same shape): [batch, frame_num, joints_feature_dim]
        the time/frequence axis=1
    computing
        1) fourier coeffs
        2) power of fft
        3) normalizing power of fft dim-wise
        4) cumsum over freq.
        5) EMD
    '''
    # 1) fourier coeffs
    pred_fourier_coeffs = np.real(np.fft.fft(pred_seqs, axis=1))
    gt_fourier_coeffs = np.real(np.fft.fft(gt_seqs, axis=1))

    # 2) power of fft
    pred_power = np.square(np.absolute(pred_fourier_coeffs))
    gt_power = np.square(np.absolute(gt_fourier_coeffs))

    # 3) normalizing power of fft dim-wise
    pred_total_power = pred_power.sum(axis=1, keepdims=True)
    gt_total_power = gt_power.sum(axis=1, keepdims=True)
    pred_norm_power = pred_power / pred_total_power
    gt_norm_power = gt_power / gt_total_power

    # 4) cumsum over freq.
    cdf_pred_power = pred_norm_power.cumsum(axis=1)
    cdf_gt_power = gt_norm_power.cumsum(axis=1)

    # 5) EMD
    emd = np.linalg.norm(cdf_pred_power - cdf_gt_power, ord=1, axis=1)
    power_weighted_emd = np.average(emd, weights=gt_total_power.squeeze())
    return power_weighted_emd

'''
def compute_npss_old(pred_seqs, gt_seqs):
    # computing 1) fourier coeffs 2)power of fft 3) normalizing power of fft dim-wise 4) cumsum over freq. 5) EMD
    gt_fourier_coeffs = np.zeros(gt_seqs.shape)
    pred_fourier_coeffs = np.zeros(pred_seqs.shape)

    # power vars
    gt_power = np.zeros((gt_fourier_coeffs.shape))
    pred_power = np.zeros((gt_fourier_coeffs.shape))

    # normalizing power vars
    gt_norm_power = np.zeros(gt_fourier_coeffs.shape)
    pred_norm_power = np.zeros(gt_fourier_coeffs.shape)

    cdf_gt_power = np.zeros(gt_norm_power.shape)
    cdf_pred_power = np.zeros(pred_norm_power.shape)

    emd = np.zeros(cdf_pred_power.shape[0:3:2])

    # used to store powers of feature_dims and sequences used for avg later
    seq_feature_power = np.zeros(gt_seqs.shape[0:3:2])
    power_weighted_emd = 0

    for s in range(gt_seqs.shape[0]):
        for d in range(gt_seqs.shape[2]):
            gt_fourier_coeffs[s, :, d] = np.fft.fft(
                gt_seqs[s, :, d])  # slice is 1D array
            pred_fourier_coeffs[s, :, d] = np.fft.fft(pred_seqs[s, :, d])

            # computing power of fft per sequence per dim
            gt_power[s, :, d] = np.square(
                np.absolute(gt_fourier_coeffs[s, :, d]))
            pred_power[s, :, d] = np.square(
                np.absolute(pred_fourier_coeffs[s, :, d]))

            # matching power of gt and pred sequences
            gt_total_power = np.sum(gt_power[s, :, d])
            pred_total_power = np.sum(pred_power[s, :, d])
            #power_diff = gt_total_power - pred_total_power

            # adding power diff to zero freq of pred seq
            #pred_power[s,0,d] = pred_power[s,0,d] + power_diff

            # computing seq_power and feature_dims power
            seq_feature_power[s, d] = gt_total_power

            # normalizing power per sequence per dim
            if gt_total_power != 0:
                gt_norm_power[s, :, d] = gt_power[s, :, d] / gt_total_power

            if pred_total_power != 0:
                pred_norm_power[s, :, d] = pred_power[s, :, d] / pred_total_power

            # computing cumsum over freq
            cdf_gt_power[s, :, d] = np.cumsum(
                gt_norm_power[s, :, d])  # slice is 1D
            cdf_pred_power[s, :, d] = np.cumsum(pred_norm_power[s, :, d])

            # computing EMD
            emd[s, d] = np.linalg.norm(
                (cdf_pred_power[s, :, d] - cdf_gt_power[s, :, d]), ord=1)

    # computing weighted emd (by sequence and feature powers)
    power_weighted_emd = np.average(emd, weights=seq_feature_power)

    return power_weighted_emd


if __name__ == "__main__":
    pred = np.random.rand(3, 4, 5)
    gt = np.random.rand(3, 4, 5)
    print(compute_npss_old(pred, gt))
    print(compute_npss(pred, gt))
'''