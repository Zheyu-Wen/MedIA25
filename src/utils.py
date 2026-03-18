import copy

import numpy as np
import nibabel as nib
from scipy import ndimage
from src import forward_op, adjoint
from sklearn.metrics import r2_score
from scipy.integrate import solve_ivp

def obj_func_p0_l2(params, alz_class):

    p0 = np.zeros([alz_class.N * 4])
    p0[:alz_class.N] = params[:alz_class.N]
    p0[alz_class.N: alz_class.N * 2] = 1 - p0[:alz_class.N]
    p0[alz_class.N * 2: alz_class.N * 3] = params[alz_class.N: alz_class.N * 2]
    p0[alz_class.N * 3: alz_class.N * 4] = 1 - p0[alz_class.N * 2: alz_class.N * 3]

    kappa_c = params[alz_class.N * 2]
    rho_c = params[alz_class.N * 2 + 1]
    gamma_c = params[alz_class.N * 2 + 2]
    rho_cb = params[alz_class.N * 2 + 3]
    kappa_b = params[alz_class.N * 2 + 4]
    rho_b = params[alz_class.N * 2 + 5]
    gamma_b = params[alz_class.N * 2 + 6]

    t = np.linspace(0, alz_class.T, alz_class.tstep)
    rhs = lambda t, s: forward_op.sys_rhs(s, t, alz_class.L, [kappa_c, rho_c, gamma_c, rho_cb, kappa_b, rho_b, gamma_b])
    s_seq = solve_ivp(rhs, [0, alz_class.T], p0, t_eval=t, method='LSODA')['y'].T

    c_abnormal_seq = s_seq[:, :alz_class.N]
    c_normal_seq = s_seq[:, alz_class.N: 2 * alz_class.N]
    b_abnormal_seq = s_seq[:, 2 * alz_class.N: 3 * alz_class.N]
    b_normal_seq = s_seq[:, 3 * alz_class.N: 4 * alz_class.N]

    '''
    Begin to solve adjoint equation. This equation will be solved in several segmentation as indicated in observation time.
    '''
    ## Below is the initialization of the adjoint equation at t=T
    if alz_class.c_obs_t[-1] == alz_class.T:
        alpha_ca1 = 1 / len(alz_class.c_obs_t) * (alz_class.d_c[-1].reshape([-1, ]) - c_abnormal_seq[-1, :].reshape([-1, ]))
    else:
        alpha_ca1 = np.zeros([alz_class.N, ])
    alpha_cn1 = np.zeros_like(alpha_ca1)
    if alz_class.b_obs_t[-1] == alz_class.T:
        alpha_ba1 = 1 / len(alz_class.b_obs_t) * (alz_class.d_b[-1].reshape([-1, ]) - b_abnormal_seq[-1, :].reshape([-1, ]))
    else:
        alpha_ba1 = np.zeros([alz_class.N, ])
    alpha_bn1 = np.zeros_like(alpha_ba1)

    alpha1 = zip(alpha_ca1, alpha_cn1, alpha_ba1, alpha_bn1)

    ## Below define the adjoint equation
    ts = np.linspace(0, alz_class.T, alz_class.tstep)
    rhs = lambda t, alpha: adjoint.adjoint_rhs(alpha, t, s_seq, ts, [kappa_c, rho_c, gamma_c, rho_cb, kappa_b, rho_b, gamma_b], alz_class)

    ## We collect all the observation time
    time_collection = list(alz_class.c_obs_t) + list(alz_class.b_obs_t)
    time_collection = list(set(time_collection))
    if alz_class.T in time_collection:
        time_collection.remove(alz_class.T)
    time_collection.append(0)
    time_collection = sorted(time_collection)[::-1]

    ## begin the solve adjoint backwardly in the for loop
    alpha_seq = np.zeros([alz_class.tstep, 4*alz_class.N])
    curr_t = alz_class.T
    t_global = np.linspace(0, alz_class.T, alz_class.tstep)
    for obs_t in time_collection:

        t = np.linspace(obs_t, curr_t, int(np.round((alz_class.tstep-1) * curr_t) - np.round((alz_class.tstep-1) * obs_t) / alz_class.T) + 1)[::-1]
        alpha_seq_tmp = solve_ivp(rhs, [curr_t, obs_t], alpha1, t_eval=t, method='LSODA')['y'].T[::-1, :]
        curr_loc = int(np.round(curr_t * (alz_class.tstep - 1)))
        obs_loc = int(np.round(obs_t * (alz_class.tstep - 1)))
        alpha_seq[obs_loc: curr_loc+1] = alpha_seq_tmp

        alpha_ca1 = alpha_seq_tmp[0, :alz_class.N]
        alpha_cn1 = alpha_seq_tmp[0, alz_class.N: 2*alz_class.N]
        alpha_ba1 = alpha_seq_tmp[0, 2*alz_class.N: 3*alz_class.N]
        alpha_bn1 = alpha_seq_tmp[0, 3*alz_class.N: 4*alz_class.N]

        if obs_t in alz_class.c_obs_t:
            loc = list(alz_class.c_obs_t).index(obs_t)
            nt_c = np.where(abs(t_global-obs_t) < alz_class.T / (2 * (alz_class.tstep - 1)))[0][0]
            alpha_ca1 += 1 / len(alz_class.c_obs_t) * (alz_class.d_c[loc].reshape([-1, ]) - c_abnormal_seq[nt_c, :].reshape([-1, ]))
            alpha_cn1 += np.zeros_like(alpha_ca1)
        if obs_t in alz_class.b_obs_t:
            loc = list(alz_class.b_obs_t).index(obs_t)
            nt_b = np.where(abs(t_global-obs_t) < alz_class.T / (2 * (alz_class.tstep - 1)))[0][0]
            alpha_ba1 += 1 / len(alz_class.b_obs_t) * (alz_class.d_b[loc].reshape([-1, ]) - b_abnormal_seq[nt_b, :].reshape([-1, ]))
            alpha_bn1 += np.zeros_like(alpha_ba1)

        alpha1 = zip(alpha_ca1, alpha_cn1, alpha_ba1, alpha_bn1)
        curr_t = obs_t

    alpha_ca_seq = alpha_seq[:, :alz_class.N]
    alpha_cn_seq = alpha_seq[:, alz_class.N: 2 * alz_class.N]
    alpha_ba_seq = alpha_seq[:, 2 * alz_class.N: 3 * alz_class.N]
    alpha_bn_seq = alpha_seq[:, 3 * alz_class.N: 4 * alz_class.N]

    obj = 0
    # rel_err = 0
    times = np.linspace(0, alz_class.T, alz_class.tstep)
    for (i, c_obs_it) in enumerate(alz_class.c_obs_t):
        nt_c_obs = np.where(abs(times-c_obs_it) < alz_class.T / (2 * (alz_class.tstep - 1)))[0][0]
        obj += 1 / (2 * len(alz_class.c_obs_t)) * (np.linalg.norm(c_abnormal_seq[nt_c_obs, :] - alz_class.d_c[i])) ** 2
    for (i, b_obs_it) in enumerate(alz_class.b_obs_t):
        nt_b_obs = np.where(abs(times-b_obs_it) < alz_class.T / (2 * (alz_class.tstep - 1)))[0][0]
        obj += 1 / (2 * len(alz_class.b_obs_t)) * (np.linalg.norm(b_abnormal_seq[nt_b_obs, :] - alz_class.d_b[i])) ** 2

    if alz_class.lambda1 != 0 and alz_class.lambda2 != 0:
        obj += alz_class.lambda1 * np.sum(np.log(1-p0[:alz_class.L.shape[0]]*alz_class.support_c+1e-16)) \
              + alz_class.lambda2 * np.sum(np.log(1-p0[2*alz_class.L.shape[0]: 3*alz_class.L.shape[0]]*alz_class.support_b+1e-16))

    ## gradient of kappa_c
    grad_kappa_c = np.sum(alpha_ca_seq * (alz_class.L @ c_abnormal_seq.T).T) * alz_class.delta_t * alz_class.kappa_c_decay
    ## gradient of rho_c
    grad_rho_c = np.dot((alpha_cn_seq - alpha_ca_seq).reshape([-1, ]), (c_abnormal_seq * c_normal_seq).reshape([-1, ])) * alz_class.delta_t * alz_class.rho_c_decay
    ## gradient of gamma_c
    grad_gamma_c = np.dot(alpha_ca_seq.reshape([-1, ]), c_abnormal_seq.reshape([-1, ])) * alz_class.delta_t * alz_class.gamma_c_decay
    ## gradient of rho_cb
    grad_rho_cb = np.dot((alpha_cn_seq - alpha_ca_seq).reshape([-1, ]), (c_normal_seq * (b_abnormal_seq)).reshape([-1, ])) * alz_class.delta_t * alz_class.rho_cb_decay
    ## gradient of kappa_b
    grad_kappa_b = np.sum(alpha_ba_seq * (alz_class.L @ b_abnormal_seq.T).T) * alz_class.delta_t * alz_class.kappa_b_decay
    ## gradient of rho_b
    grad_rho_b = np.dot((alpha_bn_seq - alpha_ba_seq).reshape([-1, ]), (b_abnormal_seq * b_normal_seq).reshape([-1, ])) * alz_class.delta_t * alz_class.rho_b_decay
    ## gradient of gamma_b
    grad_gamma_b = np.dot(alpha_ba_seq.reshape([-1, ]), b_abnormal_seq.reshape([-1, ])) * alz_class.delta_t * alz_class.gamma_b_decay
    ## gradient descent of pc0
    grad_pc0_abnormal = (-alpha_ca_seq[0, :] + alpha_cn_seq[0, :])
    if alz_class.lambda1 != 0:
        grad_pc0_abnormal += alz_class.lambda1 / (p0[:alz_class.N] - 1 + 1e-16)
    grad_pc0_abnormal *= alz_class.support_c
    ## gradient descent of pb0
    grad_pb0_abnormal = (-alpha_ba_seq[0, :] + alpha_bn_seq[0, :])
    if alz_class.lambda2 != 0:
        grad_pb0_abnormal += alz_class.lambda2 / (p0[alz_class.N * 2: alz_class.N * 3] - 1 + 1e-16)
    grad_pb0_abnormal *= alz_class.support_b

    alz_class.lambda1 *= alz_class.lambda_factor
    alz_class.lambda2 *= alz_class.lambda_factor

    return obj, np.concatenate((list(grad_pc0_abnormal), list(grad_pb0_abnormal), [grad_kappa_c], [grad_rho_c], [grad_gamma_c], [grad_rho_cb], [grad_kappa_b], [grad_rho_b], [grad_gamma_b]), axis=0)

def writeNII(img, filename, affine=None, ref_image=None):
    '''
    function to write a nifti image, creates a new nifti object
    '''
    if ref_image is not None:
        data = nib.Nifti1Image(img, affine=ref_image.affine, header=ref_image.header)
        data.header['datatype'] = 64
        data.header['glmax'] = np.max(img)
        data.header['glmin'] = np.min(img)
    elif affine is not None:
        data = nib.Nifti1Image(img, affine=affine)
    else:
        data = nib.Nifti1Image(img, np.eye(4))

    nib.save(data, filename)


def graphlaplaciandense(A,normalize=False):
    D = np.sum(A,1)
    D[np.where(D<1)]=1
    L = np.diag(D)-A
    if normalize:
        Dmoh = np.diag(D**(-1/2))
        L = Dmoh@L@Dmoh
    return L

def unzip(s):

    n = s.shape[0]//4
    tau_a = s[:n]
    tau_n = s[n: 2*n]
    abeta_a = s[2*n: 3*n]
    abeta_n = s[3*n: 4*n]
    return tau_a, tau_n, abeta_a, abeta_n

def zip(ta, tn, abetaa, abetan):

    return np.hstack([ta, tn, abetaa, abetan])

def compute_com(img):
    return tuple(int(s) for s in ndimage.center_of_mass(img))

def moving_avg_func(arr, window=5):
    result = []
    for i in range(len(arr)):
        if i < window - 1:
            result.append(arr[i])
        else:
            avg = np.mean(arr[i-4: i+1])
            result.append(avg)

    return result

def comp_delta_year(time1, time2):
    year1 = int(time1[:4])
    month1 = int(time1[5:7])
    day1 = int(time1[-2:])

    year2 = int(time2[:4])
    month2 = int(time2[5:7])
    day2 = int(time2[-2:])

    t1 = year1 * 365 + month1*30 + day1
    t2 = year2 * 365 + month2*30 + day2
    return (t2 - t1) / 365

def extract_time_list(time_str, init_age, init_time):

    if ',' in time_str:
        ans = []
        data_time_split = time_str.split(',')
        for (i_t, t) in enumerate(data_time_split):
            if i_t == len(data_time_split) - 1:
                if '\'' in t:
                    pat_time = t[2:-2]
                else:
                    pat_time = t[1:-1]
            else:
                if '\'' in t:
                    pat_time = t[2:-1]
                else:
                    pat_time = t[1:]
            ans.append(comp_delta_year(init_time, pat_time) + init_age)
        return ans
    else:
        if len(time_str) < 12:
            return []
        else:
            return [comp_delta_year(init_time, time_str[2: 12]) + init_age]


def dps_model(ages, alpha, beta, hyper_param=0.41, min_dps=-0.134, max_dps=2.318):
    t = (alpha * np.asarray(ages) + beta - min_dps + hyper_param) / (max_dps - min_dps + hyper_param)
    return t

def check_gradient(alz_class):

    s_seq_pats = alz_class.simulate_forward()

    alz_class.c_obs_t = [0.8, 0.85, 0.9, 1]
    alz_class.b_obs_t = [0.8, 0.85, 0.9, 1]

    alz_class.support_c = np.ones([alz_class.N])
    alz_class.d_c = np.zeros([len(alz_class.c_obs_t), alz_class.N])
    alz_class.support_b = np.ones([alz_class.N])
    alz_class.d_b = np.zeros([len(alz_class.b_obs_t), alz_class.N])
    alz_class.noise_level = 0

    for it in range(len(alz_class.c_obs_t)):
        alz_class.d_c[it] = s_seq_pats[int(alz_class.c_obs_t[it] * (alz_class.tstep - 1)), :alz_class.N]
        noise = np.random.randn(alz_class.L.shape[0])
        noise = noise / np.linalg.norm(noise) * np.linalg.norm(alz_class.d_c[it]) * alz_class.noise_level
        alz_class.d_c[it] += noise
    for it in range(len(alz_class.b_obs_t)):
        alz_class.d_b[it] = s_seq_pats[int(alz_class.b_obs_t[it] * (alz_class.tstep - 1)), alz_class.N * 2: alz_class.N * 3]
        noise = np.random.randn(alz_class.L.shape[0])
        noise = noise / np.linalg.norm(noise) * np.linalg.norm(alz_class.d_b[it]) * alz_class.noise_level
        alz_class.d_b[it] += noise

    curr_scalar_params = alz_class.params_true.copy() + np.random.rand(*alz_class.params_true.shape)
    curr_p0 = alz_class.p0_true.copy() + np.random.rand(*alz_class.p0_true.shape) * 0.1
    curr_p0[curr_p0>1] = 1
    curr_p0[alz_class.N: alz_class.N * 2] = 1 - curr_p0[:alz_class.N]
    curr_p0[alz_class.N * 3: alz_class.N * 4] = 1 - curr_p0[alz_class.N * 2: alz_class.N * 3]

    curr_params = np.concatenate((list(curr_p0[:alz_class.N]), list(curr_p0[2 * alz_class.N: 3 * alz_class.N]), list(curr_scalar_params)), axis=0)
    obj, total_grad = obj_func_p0_l2(curr_params, alz_class)

    alz_class.lambda1 /= alz_class.lambda_factor
    alz_class.lambda2 /= alz_class.lambda_factor

    obj_val = get_obj_val(alz_class, curr_params)

    ## check grad for pc0 of one patient
    random_posi = np.random.choice(np.arange(alz_class.N), 1)[0]
    curr_params[random_posi] += alz_class.delta_t
    obj_val_p_delta = get_obj_val(alz_class, curr_params)
    curr_params[random_posi] -= alz_class.delta_t
    num_grad = (obj_val_p_delta - obj_val) / alz_class.delta_t
    grad_err = abs(num_grad - total_grad[random_posi]) / abs(num_grad)
    print('closed form grad: {}, numerical grad: {}, relative error of grad IC: {}'.format(
        total_grad[random_posi], num_grad, grad_err))

    ## check grad for pc0 of one patient
    random_posi = np.random.choice(np.arange(alz_class.N), 1)[0]
    curr_params[random_posi + alz_class.N] += alz_class.delta_t
    obj_val_p_delta = get_obj_val(alz_class, curr_params)
    curr_params[random_posi + alz_class.N] -= alz_class.delta_t
    num_grad = (obj_val_p_delta - obj_val) / alz_class.delta_t
    grad_err = abs(num_grad - total_grad[random_posi + alz_class.N]) / abs(num_grad)
    print('closed form grad: {}, numerical grad: {}, relative error of grad IC: {}'.format(
        total_grad[random_posi + alz_class.N], num_grad, grad_err))

    ## check grad for kappa_c of one patient
    curr_params[alz_class.N * 2] += alz_class.delta_t
    obj_val_p_delta = get_obj_val(alz_class, curr_params)
    curr_params[alz_class.N * 2] -= alz_class.delta_t
    num_grad = (obj_val_p_delta - obj_val) / alz_class.delta_t
    grad_err = abs(num_grad - total_grad[alz_class.N * 2]) / abs(num_grad)
    print('closed form grad: {}, numerical grad: {}, relative error of grad kappa_c: {}'.format(total_grad[alz_class.N * 2], num_grad, grad_err))

    ## check grad for rho_c of one patient
    curr_params[alz_class.N * 2 + 1] += alz_class.delta_t
    obj_val_p_delta = get_obj_val(alz_class, curr_params)
    curr_params[alz_class.N * 2 + 1] -= alz_class.delta_t
    num_grad = (obj_val_p_delta - obj_val) / alz_class.delta_t
    grad_err = abs(num_grad - total_grad[alz_class.N * 2 + 1]) / abs(num_grad)
    print('closed form grad: {}, numerical grad: {}, relative error of grad rho_c: {}'.format(total_grad[alz_class.N * 2 + 1], num_grad, grad_err))

    ## check grad for gamma_c of one patient
    curr_params[alz_class.N * 2 + 2] += alz_class.delta_t
    obj_val_p_delta = get_obj_val(alz_class, curr_params)
    curr_params[alz_class.N * 2 + 2] -= alz_class.delta_t
    num_grad = (obj_val_p_delta - obj_val) / alz_class.delta_t
    grad_err = abs(num_grad - total_grad[alz_class.N * 2 + 2]) / abs(num_grad)
    print('closed form grad: {}, numerical grad: {}, relative error of grad gamma_c: {}'.format(total_grad[alz_class.N * 2 + 2], num_grad, grad_err))

    ## check grad for rho_cb of one patient
    curr_params[alz_class.N * 2 + 3] += alz_class.delta_t
    obj_val_p_delta = get_obj_val(alz_class, curr_params)
    curr_params[alz_class.N * 2 + 3] -= alz_class.delta_t
    num_grad = (obj_val_p_delta - obj_val) / alz_class.delta_t
    grad_err = abs(num_grad - total_grad[alz_class.N * 2 + 3]) / abs(num_grad)
    print('closed form grad: {}, numerical grad: {}, relative error of grad rho_cb: {}'.format(total_grad[alz_class.N * 2 + 3], num_grad, grad_err))

    ## check grad for kappa_b of one patient
    curr_params[alz_class.N * 2 + 4] += alz_class.delta_t
    obj_val_p_delta = get_obj_val(alz_class, curr_params)
    curr_params[alz_class.N * 2 + 4] -= alz_class.delta_t
    num_grad = (obj_val_p_delta - obj_val) / alz_class.delta_t
    grad_err = abs(num_grad - total_grad[alz_class.N * 2 + 4]) / abs(num_grad)
    print('closed form grad: {}, numerical grad: {}, relative error of grad kappa_b: {}'.format(total_grad[alz_class.N * 2 + 4], num_grad, grad_err))

    ## check grad for rho_b of one patient
    curr_params[alz_class.N * 2 + 5] += alz_class.delta_t
    obj_val_p_delta = get_obj_val(alz_class, curr_params)
    curr_params[alz_class.N * 2 + 5] -= alz_class.delta_t
    num_grad = (obj_val_p_delta - obj_val) / alz_class.delta_t
    grad_err = abs(num_grad - total_grad[alz_class.N * 2 + 5]) / abs(num_grad)
    print('closed form grad: {}, numerical grad: {}, relative error of grad rho_b: {}'.format(total_grad[alz_class.N * 2 + 5], num_grad, grad_err))

    ## check grad for gamma_b of one patient
    curr_params[alz_class.N * 2 + 6] += alz_class.delta_t
    obj_val_p_delta = get_obj_val(alz_class, curr_params)
    curr_params[alz_class.N * 2 + 6] -= alz_class.delta_t
    num_grad = (obj_val_p_delta - obj_val) / alz_class.delta_t
    grad_err = abs(num_grad - total_grad[alz_class.N * 2 + 6]) / abs(num_grad)
    print('closed form grad: {}, numerical grad: {}, relative error of grad gamma_b: {}'.format(total_grad[alz_class.N * 2 + 6], num_grad, grad_err))

def get_obj_val(alz_class, curr_params):

    p0 = np.zeros([alz_class.N * 4])
    p0[:alz_class.N] = curr_params[:alz_class.N]
    p0[alz_class.N: alz_class.N * 2] = 1 - p0[:alz_class.N]
    p0[alz_class.N * 2: alz_class.N * 3] = curr_params[alz_class.N: alz_class.N * 2]
    p0[alz_class.N * 3: alz_class.N * 4] = 1 - p0[alz_class.N * 2: alz_class.N * 3]

    t = np.linspace(0, alz_class.T, alz_class.tstep)

    obj = 0

    rhs = lambda t, s: forward_op.sys_rhs(s, t, alz_class.L, curr_params[alz_class.N * 2: alz_class.N * 2 + 7])
    s_seq = solve_ivp(rhs, [0, alz_class.T], p0, t_eval=t, method='LSODA')['y'].T

    c_abnormal_seq = s_seq[:, :alz_class.N]
    b_abnormal_seq = s_seq[:, 2 * alz_class.N: 3 * alz_class.N]

    times = np.linspace(0, alz_class.T, alz_class.tstep)
    for (i, c_obs_it) in enumerate(alz_class.c_obs_t):
        nt_c_obs = np.where(abs(times - c_obs_it) < alz_class.T / (2 * (alz_class.tstep - 1)))[0][0]
        obj += 1 / (2 * len(alz_class.c_obs_t)) * (np.linalg.norm(c_abnormal_seq[nt_c_obs, :] - alz_class.d_c[i])) ** 2
    for (i, b_obs_it) in enumerate(alz_class.b_obs_t):
        nt_b_obs = np.where(abs(times - b_obs_it) < alz_class.T / (2 * (alz_class.tstep - 1)))[0][0]
        obj += 1 / (2 * len(alz_class.b_obs_t)) * (np.linalg.norm(b_abnormal_seq[nt_b_obs, :] - alz_class.d_b[i])) ** 2

    return obj