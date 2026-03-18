from src import utils
import numpy as np

def adjoint_rhs(alpha, t, s_seq, times, theta, alz_class):

    alpha_ca, alpha_cn, alpha_ba, alpha_bn = utils.unzip(alpha)
    kappa_c, rho_c, gamma_c, rho_cb, kappa_b, rho_b, gamma_b = theta
    if t < 0:
        nt = 0
    elif t > np.max(times):
        nt = s_seq.shape[0] - 1
    else:
        nt = np.where(abs(times-t) < alz_class.T / (2 * (alz_class.tstep - 1)))[0][0]

    c_abnormal_seq = s_seq[:, :alz_class.L.shape[0]]
    c_normal_seq = s_seq[:, alz_class.L.shape[0]: 2 * alz_class.L.shape[0]]
    b_abnormal_seq = s_seq[:, 2 * alz_class.L.shape[0]: 3 * alz_class.L.shape[0]]
    b_normal_seq = s_seq[:, 3 * alz_class.L.shape[0]: 4 * alz_class.L.shape[0]]

    out_ca = kappa_c * alz_class.L @ alpha_ca \
             + rho_c * c_normal_seq[nt] * (alpha_cn - alpha_ca) \
             + gamma_c * alpha_ca

    out_cn = rho_c * c_abnormal_seq[nt] * (alpha_cn - alpha_ca) + rho_cb * b_abnormal_seq[nt] * (alpha_cn - alpha_ca)

    out_ba = kappa_b * alz_class.L @ alpha_ba \
             + rho_b * b_normal_seq[nt] * (alpha_bn - alpha_ba) + gamma_b * alpha_ba \
             + rho_cb * c_normal_seq[nt] * (alpha_cn - alpha_ca)
    out_bn = rho_b * b_abnormal_seq[nt] * (alpha_bn - alpha_ba)

    out = utils.zip(out_ca, out_cn, out_ba, out_bn)

    return out
