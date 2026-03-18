from src import utils



def sys_rhs(s, t, L, theta):

    kappa_c, rho_c, gamma_c, rho_cb, kappa_b, rho_b, gamma_b = theta
    tau_a, tau_n, abeta_a, abeta_n = utils.unzip(s)
    rhs_ta  = - kappa_c*(L@tau_a) + rho_c*tau_a*tau_n - gamma_c*tau_a + rho_cb * abeta_a * tau_n
    rhs_tn  = - rho_c*tau_a*tau_n - rho_cb * abeta_a * tau_n
    rhs_ba  = - kappa_b*(L@abeta_a) + rho_b*abeta_a*abeta_n - gamma_b*abeta_a
    rhs_bn  = - rho_b*abeta_a*abeta_n

    return utils.zip(rhs_ta, rhs_tn, rhs_ba, rhs_bn)