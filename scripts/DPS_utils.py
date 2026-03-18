import numpy as np
import scipy.optimize as sci_opti


def mental_score_model(ages, alpha, beta, theta):

    age_mask = np.zeros_like(ages)
    age_mask[ages>0] = 1
    dps = (alpha.reshape([1, -1, 1]) * ages + beta.reshape([1, -1, 1])) * age_mask
    M = 1 + 1/(theta[:, 0].reshape([-1, 1, 1]) + 1e-8) * np.exp(-theta[:, 1].reshape([-1, 1, 1])/(theta[:, 0].reshape([-1, 1, 1]) + 1e-8)*(dps - theta[:, 2].reshape([-1, 1, 1])))
    M = M ** (-theta[:, 0].reshape([-1, 1, 1]) + 1e-8) * age_mask
    return M, dps

def obj_DPS(M, ages, params_guess):

    n_mental_score, n_pat = ages.shape[0], ages.shape[1]
    alpha_guess = params_guess[:n_pat]
    beta_guess = params_guess[n_pat: 2 * n_pat]

    theta_guess = np.zeros([n_mental_score, 3])
    for i in range(n_mental_score):
        theta_guess[i] = params_guess[2*n_pat + i*3: 2*n_pat + (i+1)*3]
    Mhat, tau_hat = mental_score_model(ages, alpha_guess, beta_guess, theta_guess)

    age_mask = np.zeros_like(ages)
    age_mask[ages > 0] = 1

    obj = 1/2 * np.linalg.norm(M.reshape([-1, ]) - (age_mask * Mhat).reshape([-1, ]))**2

    grad_logistic = 1 + 1 / (theta_guess[:, 0].reshape([-1, 1, 1]) + 1e-8) * np.exp(-theta_guess[:, 1].reshape([-1, 1, 1]) / (theta_guess[:, 0].reshape([-1, 1, 1]) + 1e-8) * (tau_hat - theta_guess[:, 2].reshape([-1, 1, 1])))
    grad_logistic = grad_logistic ** (-theta_guess[:, 0].reshape([-1, 1, 1]) - 1 + 1e-8)

    grad_theta0 = age_mask * (Mhat - M) * Mhat * (-np.log(1 + 1/(theta_guess[:, 0].reshape([-1, 1, 1]) + 1e-8) * np.exp(-theta_guess[:, 1].reshape([-1, 1, 1]) / (theta_guess[:, 0].reshape([-1, 1, 1]) + 1e-8) * (tau_hat - theta_guess[:, 2].reshape([-1, 1, 1])))) - theta_guess[:, 0].reshape([-1, 1, 1])
                                                  / (1 + 1/(theta_guess[:, 0].reshape([-1, 1, 1]) + 1e-8) * np.exp(-theta_guess[:, 1].reshape([-1, 1, 1]) / (theta_guess[:, 0].reshape([-1, 1, 1]) + 1e-8) * (tau_hat - theta_guess[:, 2].reshape([-1, 1, 1]))))
                                                  * 1/(theta_guess[:, 0].reshape([-1, 1, 1])**2 + 1e-8) * np.exp(-theta_guess[:, 1].reshape([-1, 1, 1])/(theta_guess[:, 0].reshape([-1, 1, 1]) + 1e-8)*(tau_hat-theta_guess[:, 2].reshape([-1, 1, 1]))) * (- 1 + theta_guess[:, 1].reshape([-1, 1, 1])/(theta_guess[:, 0].reshape([-1, 1, 1]) + 1e-8)*(tau_hat - theta_guess[:, 2].reshape([-1, 1, 1]))))
    grad_theta1 = age_mask * (Mhat - M) * grad_logistic * 1/(theta_guess[:, 0].reshape([-1, 1, 1])**2 + 1e-8) * np.exp(-theta_guess[:, 1].reshape([-1, 1, 1])
                                                  /(theta_guess[:, 0].reshape([-1, 1, 1]) + 1e-8)*(tau_hat-theta_guess[:, 2].reshape([-1, 1, 1]))) * (tau_hat - theta_guess[:, 2].reshape([-1, 1, 1]))
    grad_theta2 = age_mask * (Mhat - M) * grad_logistic * -theta_guess[:, 1].reshape([-1, 1, 1]) / (theta_guess[:, 0].reshape([-1, 1, 1]) + 1e-8) \
                  * np.exp(-theta_guess[:, 1].reshape([-1, 1, 1])/(theta_guess[:, 0].reshape([-1, 1, 1]) + 1e-8)*(tau_hat-theta_guess[:, 2].reshape([-1, 1, 1])))

    grad_alpha = age_mask * (Mhat - M) * grad_logistic * (theta_guess[:, 1].reshape([-1, 1, 1])/(theta_guess[:, 0].reshape([-1, 1, 1]) + 1e-8)) \
                 * np.exp(-theta_guess[:, 1].reshape([-1, 1, 1])/(theta_guess[:, 0].reshape([-1, 1, 1]) + 1e-8)*(tau_hat - theta_guess[:, 2].reshape([-1, 1, 1]))) \
                 * ages
    grad_beta = age_mask * (Mhat - M) * grad_logistic * (theta_guess[:, 1].reshape([-1, 1, 1])/(theta_guess[:, 0].reshape([-1, 1, 1]) + 1e-8)) \
                * np.exp(-theta_guess[:, 1].reshape([-1, 1, 1])/(theta_guess[:, 0].reshape([-1, 1, 1]) + 1e-8)*(tau_hat - theta_guess[:, 2].reshape([-1, 1, 1]))) \
                * 1


    grad_theta = np.zeros([n_mental_score * 3, ])
    for i in range(n_mental_score):
        grad_theta[i * 3] = np.sum(grad_theta0[i])
        grad_theta[i * 3 + 1] = np.sum(grad_theta1[i])
        grad_theta[i * 3 + 2] = np.sum(grad_theta2[i])

    grad_alpha = np.sum(np.sum(grad_alpha, axis=0), axis=1)
    grad_beta = np.sum(np.sum(grad_beta, axis=0), axis=1)

    return obj, np.concatenate((grad_alpha, grad_beta, grad_theta), axis=0)

def check_gradient():

    n_pat = 100
    n_mental_score = 1
    theta = np.asarray([0.5, 0.5, 0.5]).reshape([1, 3]) + np.random.rand(1, 3)
    ages = (np.random.rand(n_pat, 3) * 100).reshape([1, n_pat, 3])
    alpha = np.random.rand(n_pat)
    beta = np.random.rand(n_pat)
    M, tau = mental_score_model(ages, alpha, beta, theta)
    age_mask = np.zeros_like(ages)
    age_mask[ages > 0] = 1

    ## check theta
    theta_guess = np.asarray([1, 1, 1]).reshape([1, 3])
    Mhat, tau_hat = mental_score_model(ages, alpha, beta, theta_guess)
    grad_logistic = 1 + 1 / (theta_guess[:, 0].reshape([-1, 1, 1]) + 1e-8) * np.exp(-theta_guess[:, 1].reshape([-1, 1, 1]) / (theta_guess[:, 0].reshape([-1, 1, 1]) + 1e-8) * (tau_hat - theta_guess[:, 2].reshape([-1, 1, 1])))
    grad_logistic = grad_logistic ** (-theta_guess[:, 0].reshape([-1, 1, 1]) - 1)

    grad_theta0 = age_mask * (Mhat - M) * Mhat * (-np.log(1 + 1/theta_guess[:, 0].reshape([-1, 1, 1]) * np.exp(-theta_guess[:, 1].reshape([-1, 1, 1]) / theta_guess[:, 0].reshape([-1, 1, 1]) * (tau_hat - theta_guess[:, 2].reshape([-1, 1, 1])))) - theta_guess[:, 0].reshape([-1, 1, 1])
                           / (1 + 1/theta_guess[:, 0].reshape([-1, 1, 1]) * np.exp(-theta_guess[:, 1].reshape([-1, 1, 1]) / theta_guess[:, 0].reshape([-1, 1, 1]) * (tau_hat - theta_guess[:, 2].reshape([-1, 1, 1]))))
                           * 1/(theta_guess[:, 0].reshape([-1, 1, 1])**2 + 1e-8) * np.exp(-theta_guess[:, 1].reshape([-1, 1, 1])
                           /(theta_guess[:, 0].reshape([-1, 1, 1]) + 1e-8)*(tau_hat-theta_guess[:, 2].reshape([-1, 1, 1]))) * (- 1 + theta_guess[:, 1].reshape([-1, 1, 1])/(theta_guess[:, 0].reshape([-1, 1, 1]) + 1e-8)*(tau_hat - theta_guess[:, 2].reshape([-1, 1, 1]))))
    grad_theta1 = age_mask * (Mhat - M) * grad_logistic * 1/(theta_guess[:, 0].reshape([-1, 1, 1])**2 + 1e-8) * np.exp(-theta_guess[:, 1].reshape([-1, 1, 1])
                                                                                                                       /(theta_guess[:, 0].reshape([-1, 1, 1]) + 1e-8)*(tau_hat-theta_guess[:, 2].reshape([-1, 1, 1]))) * (tau_hat - theta_guess[:, 2].reshape([-1, 1, 1]))
    grad_theta2 = age_mask * (Mhat - M) * grad_logistic * -theta_guess[:, 1].reshape([-1, 1, 1]) / (theta_guess[:, 0].reshape([-1, 1, 1]) + 1e-8) \
                  * np.exp(-theta_guess[:, 1].reshape([-1, 1, 1])/(theta_guess[:, 0].reshape([-1, 1, 1]) + 1e-8)*(tau_hat-theta_guess[:, 2].reshape([-1, 1, 1])))

    grad_theta = np.zeros([3, ])
    grad_theta[0] = np.sum(grad_theta0[0])
    grad_theta[1] = np.sum(grad_theta1[0])
    grad_theta[2] = np.sum(grad_theta2[0])

    delta_theta = 1e-10
    theta_guess0 = np.asarray([1 + delta_theta, 1, 1]).reshape([1, 3])
    Mhat0, _ = mental_score_model(ages, alpha, beta, theta_guess0)
    grad_theta0_num = 1/2 * (np.linalg.norm(Mhat0 - M)**2 - np.linalg.norm(Mhat - M)**2) / delta_theta
    print('theta 0 grad error: ', abs(grad_theta[0] - grad_theta0_num) / abs(grad_theta0_num))

    theta_guess1 = np.asarray([1, 1 + delta_theta, 1]).reshape([1, 3])
    Mhat1, _ = mental_score_model(ages, alpha, beta, theta_guess1)
    grad_theta1_num = 1/2 * (np.linalg.norm(Mhat1 - M)**2 - np.linalg.norm(Mhat - M)**2) / delta_theta
    print('theta 1 grad error: ', abs(grad_theta[1] - grad_theta1_num) / abs(grad_theta1_num))

    theta_guess2 = np.asarray([1, 1, 1 + delta_theta]).reshape([1, 3])
    Mhat2, _ = mental_score_model(ages, alpha, beta, theta_guess2)
    grad_theta2_num = 1/2 * (np.linalg.norm(Mhat2 - M)**2 - np.linalg.norm(Mhat - M)**2) / delta_theta
    print('theta 2 grad error: ', abs(grad_theta[2] - grad_theta2_num) / abs(grad_theta2_num))

    ## check alpha and beta
    alpha_guess = np.random.rand(n_pat) * 0.01 #np.zeros([n_pat, ])
    beta_guess = np.random.rand(n_pat) * 0.01 #np.zeros([n_pat, ])
    Mhat, tau_hat = mental_score_model(ages, alpha_guess, beta_guess, theta)
    grad_logistic = 1 + 1 / (theta[:, 0].reshape([-1, 1, 1]) + 1e-8) * np.exp(-theta[:, 1].reshape([-1, 1, 1]) / (theta[:, 0].reshape([-1, 1, 1]) + 1e-8) * (tau_hat - theta[:, 2].reshape([-1, 1, 1])))
    grad_logistic = grad_logistic ** (-theta[:, 0].reshape([-1, 1, 1]) - 1)
    grad_alpha = age_mask * (Mhat - M) * grad_logistic * (theta[:, 1].reshape([-1, 1, 1])/(theta[:, 0].reshape([-1, 1, 1]) + 1e-8)) \
                 * np.exp(-theta[:, 1].reshape([-1, 1, 1])/(theta[:, 0].reshape([-1, 1, 1]) + 1e-8)*(tau_hat - theta[:, 2].reshape([-1, 1, 1]))) \
                 * (ages*np.exp(-(alpha_guess.reshape([1, -1, 1])*ages + beta_guess.reshape([1, -1, 1])))) / (1 + np.exp(-(alpha_guess.reshape([1, -1, 1])*ages + beta_guess.reshape([1, -1, 1]))))**2
    grad_beta = age_mask * (Mhat - M) * grad_logistic * (theta[:, 1].reshape([-1, 1, 1])/(theta[:, 0].reshape([-1, 1, 1]) + 1e-8))\
                 * np.exp(-theta[:, 1].reshape([-1, 1, 1])/(theta[:, 0].reshape([-1, 1, 1]) + 1e-8)*(tau_hat - theta[:, 2].reshape([-1, 1, 1]))) \
                 * (np.exp(-(alpha_guess.reshape([1, -1, 1])*ages + beta_guess.reshape([1, -1, 1])))) / (1 + np.exp(-(alpha_guess.reshape([1, -1, 1])*ages + beta_guess.reshape([1, -1, 1]))))**2
    grad_alpha = np.sum(np.sum(grad_alpha, axis=0), axis=1)
    grad_beta = np.sum(np.sum(grad_beta, axis=0), axis=1)

    delta_theta = 1e-5
    alpha_guess_ = alpha_guess.copy()
    alpha_guess_[0] = alpha_guess[0] + delta_theta # * np.ones_like(alpha_guess)
    Mhat_alpha, _ = mental_score_model(ages, alpha_guess_, beta_guess, theta)
    grad_alpha_num = 1 / 2 * (np.linalg.norm(Mhat_alpha[:, 0, :] - M[:, 0, :])**2 - np.linalg.norm(Mhat[:, 0, :] - M[:, 0, :])**2) / delta_theta
    print('alpha grad error: ', np.linalg.norm(grad_alpha[0] - grad_alpha_num) / np.linalg.norm(grad_alpha_num) + 1e-8)

    delta_theta = 1e-5
    beta_guess_ = beta_guess + delta_theta * np.ones_like(beta_guess)
    Mhat_beta, _ = mental_score_model(ages, alpha_guess, beta_guess_, theta)
    grad_beta_num = 1 / 2 * (np.linalg.norm(Mhat_beta - M, axis=(0, 2))**2 - np.linalg.norm(Mhat - M, axis=(0, 2))**2) / delta_theta
    print('beta grad error: ', np.linalg.norm(grad_beta - grad_beta_num) / np.linalg.norm(grad_beta_num) + 1e-8)


def synthetic_test():

    n_mental_score = 5
    n_pat = 100

    theta_scaling = 0.5
    theta = 1 - np.random.rand(n_mental_score, 3) * theta_scaling
    ages = (1 - 0.5 * np.random.rand(n_mental_score, n_pat, 3)) * 100

    scaling = 0.05
    alpha = np.random.rand(n_pat) * scaling
    beta = np.random.rand(n_pat) * scaling
    M, tau = mental_score_model(ages, alpha, beta, theta)

    theta_guess = np.random.rand(n_mental_score, 3)
    vec_theta_guess = np.zeros([n_mental_score * 3, ])
    for i in range(n_mental_score):
        vec_theta_guess[i * 3] = theta_guess[i, 0]
        vec_theta_guess[i * 3 + 1] = theta_guess[i, 1]
        vec_theta_guess[i * 3 + 2] = theta_guess[i, 2]

    alpha_guess = np.random.rand(n_pat) * scaling
    beta_guess = np.random.rand(n_pat) * scaling

    objective_func = lambda params: obj_DPS(M, ages, params)
    params = np.concatenate((alpha_guess, beta_guess, vec_theta_guess), axis=0)
    bounds = []
    for i in range(2 * n_pat):
        bounds.append(tuple([0, scaling]))

    for i in range(3*n_mental_score):
        bounds.append(tuple([1 - theta_scaling, 1 + theta_scaling]))
    params_est, fmin, Dict = sci_opti.fmin_l_bfgs_b(objective_func, params, bounds=bounds)

    alpha_guess, beta_guess, vec_theta_guess = params_est[:n_pat], params_est[n_pat: 2*n_pat], params_est[2*n_pat:]
    for i in range(n_mental_score):
        theta_guess[i, 0] = vec_theta_guess[i * 3]
        theta_guess[i, 1] = vec_theta_guess[i * 3 + 1]
        theta_guess[i, 2] = vec_theta_guess[i * 3 + 2]
    Mhat, tau_hat = mental_score_model(ages, alpha_guess, beta_guess, theta_guess)

    print('M error: {}, theta error: {}, alpha error: {}'
              .format(np.linalg.norm(M - Mhat) / np.linalg.norm(M),
                      np.linalg.norm(theta_guess - theta) / np.linalg.norm(theta),
                      np.linalg.norm(alpha_guess - alpha) / np.linalg.norm(alpha)))
    print('stop')

if __name__ == '__main__':
    check_gradient()