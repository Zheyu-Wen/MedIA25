import copy
import numpy as np
import scipy.optimize as sci_opti
from src import forward_op, adjoint, utils
from scipy.integrate import solve_ivp
from sklearn.metrics import r2_score

def inverse_p0_parameters(alz_class):

    ## Allowed number of non-zero values in IC
    n_posi_tau = alz_class.tau_sparsity
    n_posi_abeta = alz_class.abeta_sparsity

    bounds = []
    for i in range(2 * alz_class.N):
        bounds.append(tuple([0, 1]))
    for i in range(7):
        bounds.append(tuple([0, float('inf')]))

    objective_func = lambda params: utils.obj_func_p0_l2(params, alz_class)

    curr_window_size_tau = alz_class.N
    curr_window_size_abeta = alz_class.N
    num_iter = 0

    while(curr_window_size_tau > n_posi_tau or curr_window_size_abeta > n_posi_abeta or num_iter < 5):

        num_iter += 1

        alz_class.support_c = np.ones([alz_class.N, ])
        alz_class.support_b = np.ones([alz_class.N, ])
        pc0_abnormal = alz_class.p0[:alz_class.N]
        pb0_abnormal = alz_class.p0[2 * alz_class.N: 3 *alz_class.N]

        params = np.concatenate((list(pc0_abnormal), list(pb0_abnormal), list(alz_class.params)), axis=0)
        params_est, fmin, Dict = sci_opti.fmin_l_bfgs_b(objective_func, params, bounds=bounds, approx_grad=False, maxls=50, pgtol=alz_class.grad_tol, maxiter=alz_class.lbfgs_maxiter)
        alz_class.p0[:alz_class.N] = params_est[:alz_class.N]
        alz_class.p0[alz_class.N: 2 * alz_class.N] = 1 -  alz_class.p0[:alz_class.N]
        alz_class.p0[2 * alz_class.N: 3 * alz_class.N] = params_est[alz_class.N: 2 * alz_class.N]
        alz_class.p0[3 * alz_class.N: 4 * alz_class.N] = 1 - alz_class.p0[2 * alz_class.N: 3 * alz_class.N]

        alz_class.params = params_est[alz_class.N * 2: alz_class.N * 2 + 7]

        alz_class.support_c = np.zeros([alz_class.N, ])
        alz_class.support_b = np.zeros([alz_class.N, ])
        pc0_abnormal = copy.deepcopy(alz_class.p0[:alz_class.N])
        pb0_abnormal = copy.deepcopy(alz_class.p0[2 * alz_class.N: 3 * alz_class.N])
        for ii in range(np.maximum(n_posi_tau, curr_window_size_tau)):
            posi_found = np.argmax(pc0_abnormal)
            alz_class.support_c[posi_found] = 1
            pc0_abnormal[posi_found] = 0

        for ii in range(np.maximum(n_posi_abeta, curr_window_size_abeta)):
            posi_found = np.argmax(pb0_abnormal)
            alz_class.support_b[posi_found] = 1
            pb0_abnormal[posi_found] = 0

        curr_window_size_tau = int(np.floor(alz_class.beta2 * curr_window_size_tau))
        curr_window_size_tau = max(curr_window_size_tau, n_posi_tau)

        curr_window_size_abeta = int(np.floor(alz_class.beta2 * curr_window_size_abeta))
        curr_window_size_abeta = max(curr_window_size_abeta, n_posi_abeta)

        alz_class.p0[:alz_class.N] *= alz_class.support_c
        alz_class.p0[alz_class.N: 2 * alz_class.N] = 1 - alz_class.p0[:alz_class.N]

        alz_class.p0[2 * alz_class.N: 3 * alz_class.N] *= alz_class.support_b
        alz_class.p0[3 * alz_class.N: 4 * alz_class.N] = 1 - alz_class.p0[2 * alz_class.N: 3 * alz_class.N]

        pc0_abnormal = alz_class.p0[:alz_class.N]
        pb0_abnormal = alz_class.p0[2 * alz_class.N: 3 * alz_class.N]
        params = np.concatenate((list(pc0_abnormal), list(pb0_abnormal), list(alz_class.params)), axis=0)

        params_est, fmin, Dict = sci_opti.fmin_l_bfgs_b(objective_func, params, bounds=bounds, approx_grad=False, maxls=50, pgtol=alz_class.grad_tol, maxiter=alz_class.lbfgs_maxiter)
        alz_class.p0[:alz_class.N] = params_est[:alz_class.N]
        alz_class.p0[alz_class.N: 2 * alz_class.N] = 1 -  alz_class.p0[:alz_class.N]
        alz_class.p0[2 * alz_class.N: 3 * alz_class.N] = params_est[alz_class.N: 2 * alz_class.N]
        alz_class.p0[3 * alz_class.N: 4 * alz_class.N] = 1 - alz_class.p0[2 * alz_class.N: 3 * alz_class.N]

        alz_class.params = params_est[alz_class.N * 2: alz_class.N * 2 + 7]

    return



