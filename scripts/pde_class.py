import numpy as np
import pandas as pd
import scipy.io as sio
from collections import defaultdict
from src import utils, forward_op
import os
from sklearn.metrics import r2_score
from scripts import inversion
from scipy.integrate import solve_ivp
import copy
import time

class Alzh_model_class:

    def __init__(self, project_path):

        ## load connectivity matrix and generate graph laplacian
        self.project_path = project_path
        self.conn_mtx = np.array(sio.loadmat(os.path.join(self.project_path, 'avg_conn_mat_normalized_sum_sym.mat'))['data'])

        self.skip_nroi = 0
        self.conn_mtx = self.conn_mtx[self.skip_nroi:, :][:, self.skip_nroi:]

        np.fill_diagonal(self.conn_mtx, 0)
        self.L = utils.graphlaplaciandense(self.conn_mtx, normalize=True)

        self.N = self.L.shape[0]

        ## time horizon and discrete time step in odeint
        self.T = 1
        self.tstep = 201
        self.delta_t = self.T / (self.tstep - 1)
        self.epsilon_ = 1e-5
        self.grad_tol = 1e-5
        self.lambda1 = 0
        self.lambda2 = 0
        self.beta2 = 0.3
        self.lambda_factor = 0.1

        self.normalize_IC = False
        self.noise_level = 0
        self.lbfgs_maxiter = 15000

        self.role = 'fitting'
        self.use_multiscan = 1
        self.save_result = 1

        self.template_id = 4

        self.kappa_c_decay = 1
        self.rho_c_decay = 1
        self.gamma_c_decay = 1
        self.rho_cb_decay = 1
        self.kappa_b_decay = 1
        self.rho_b_decay = 1
        self.gamma_b_decay = 1
        self.remove_param_loc = 10
        self.ablation_param = 'none'



    def simulate_forward(self):


        self.mu = np.asarray([
            [np.log(0.7), np.log(5), np.log(0.8), np.log(1), np.log(1), np.log(4), np.log(1)],
            [np.log(0.3), np.log(3), np.log(0.1), np.log(2), np.log(0.5), np.log(3), np.log(0.1)],
            [np.log(0.5), np.log(3), np.log(0.1), np.log(0.5), np.log(0.8), np.log(2), np.log(0.2)]
            ])
        self.sigma = np.asarray([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                                 [1,   1,   1,   1,   1,   1,   1],
                                 [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]]) * 0.2

        self.mu    = self.mu[0, :]
        self.sigma = self.sigma[0, :]

        self.params_true = np.zeros([self.mu.shape[0]])

        self.p0_true = np.zeros([self.N * 4])
        self.p0_true[66] = 0.5
        self.p0_true[68] = 0.5
        self.p0_true[self.N: self.N * 2] = 1 - self.p0_true[:self.N]

        self.p0_true[66 + self.N * 2] = 1
        self.p0_true[68 + self.N * 2] = 1
        self.p0_true[self.N * 3: self.N * 4] = 1 - self.p0_true[self.N * 2: self.N * 3]

        kappa_c = np.exp(np.random.randn() * self.sigma[0] + self.mu[0])
        rho_c = np.exp(np.random.randn() * self.sigma[1] + self.mu[1])
        gamma_c = np.exp(np.random.randn() * self.sigma[2] + self.mu[2])
        rho_cb = np.exp(np.random.randn() * self.sigma[3] + self.mu[3])
        kappa_b = np.exp(np.random.randn() * self.sigma[4] + self.mu[4])
        rho_b = np.exp(np.random.randn() * self.sigma[5] + self.mu[5])
        gamma_b = np.exp(np.random.randn() * self.sigma[6] + self.mu[6])

        self.params_true[0] = copy.deepcopy(kappa_c)
        self.params_true[1] = copy.deepcopy(rho_c)
        self.params_true[2] = copy.deepcopy(gamma_c)
        self.params_true[3] = copy.deepcopy(rho_cb)
        self.params_true[4] = copy.deepcopy(kappa_b)
        self.params_true[5] = copy.deepcopy(rho_b)
        self.params_true[6] = copy.deepcopy(gamma_b)

        rhs = lambda t, s: forward_op.sys_rhs(s, t, self.L, list(self.params_true))
        t = np.linspace(0, self.T, self.tstep)
        s_seq = solve_ivp(rhs, [0, self.T], self.p0_true, t_eval=t, method='LSODA')['y'].T
        obs_data = s_seq
        
        return obs_data

    def forward_ablation_study(self):

        ## each time, we set one of the parameters to be zero

        self.params_true = np.zeros([7, ])

        self.p0_true = np.zeros([self.N * 4])
        self.p0_true[66] = 1
        self.p0_true[68] = 1
        self.p0_true[self.N: self.N * 2] = 1 - self.p0_true[:self.N]

        self.p0_true[66 + self.N * 2] = 1
        self.p0_true[68 + self.N * 2] = 1
        self.p0_true[self.N * 3: self.N * 4] = 1 - self.p0_true[self.N * 2: self.N * 3]

        kappa_c = 5
        rho_c = 20
        gamma_c = 10
        rho_cb = 20
        kappa_b = 8
        rho_b = 20
        gamma_b = 10

        ablation_output = np.zeros([8, self.tstep, 456])
        self.T = 5

        for iablation in range(7):

            self.params_true[0] = copy.deepcopy(kappa_c)
            self.params_true[1] = copy.deepcopy(rho_c)
            self.params_true[2] = copy.deepcopy(gamma_c)
            self.params_true[3] = copy.deepcopy(rho_cb)
            self.params_true[4] = copy.deepcopy(kappa_b)
            self.params_true[5] = copy.deepcopy(rho_b)
            self.params_true[6] = copy.deepcopy(gamma_b)

            self.params_true[iablation] = 0
            rhs = lambda t, s: forward_op.sys_rhs(s, t, self.L, list(self.params_true))
            t = np.linspace(0, self.T, self.tstep)
            s_seq = solve_ivp(rhs, [0, self.T], self.p0_true, t_eval=t, method='LSODA')['y'].T
            ablation_output[iablation] = s_seq

        self.params_true[0] = copy.deepcopy(kappa_c)
        self.params_true[1] = copy.deepcopy(rho_c)
        self.params_true[2] = copy.deepcopy(gamma_c)
        self.params_true[3] = copy.deepcopy(rho_cb)
        self.params_true[4] = copy.deepcopy(kappa_b)
        self.params_true[5] = copy.deepcopy(rho_b)
        self.params_true[6] = copy.deepcopy(gamma_b)
        rhs = lambda t, s: forward_op.sys_rhs(s, t, self.L, list(self.params_true))
        t = np.linspace(0, self.T, self.tstep)
        s_seq = solve_ivp(rhs, [0, self.T], self.p0_true, t_eval=t, method='LSODA')['y'].T
        ablation_output[-1] = s_seq

        np.save(f'{self.save_folder}/MedIA_ablation_study.npy', ablation_output)


    def synthetic_inversion(self):

        self.grad_tol = 1e-5
        self.tau_sparsity = 5
        self.abeta_sparsity = 10

        nrepeat = 100

        four_obs = [0.8, 0.85, 0.9, 0.95]
        one_obs = [0.95]

        for obs_t in [one_obs, four_obs]:

            self.c_obs_t = obs_t
            self.b_obs_t = obs_t

            true_param = np.zeros([nrepeat, 7])
            true_p0 = np.zeros([nrepeat, self.N * 4])
            est_param = np.zeros([nrepeat, 7])
            est_p0 = np.zeros([nrepeat, self.N * 4])

            for irepeat in range(nrepeat):

                obs_data = self.simulate_forward()

                self.d_c = np.zeros([len(self.c_obs_t), self.N])
                for itime in range(len(self.c_obs_t)):
                    self.d_c[itime] = obs_data[int(self.c_obs_t[itime] * (self.tstep-1)), :self.N]
                    noise = np.random.randn(self.N)
                    noise = noise / np.linalg.norm(noise) * np.linalg.norm(self.d_c[itime]) * self.noise_level
                    self.d_c[itime] += noise

                self.d_b = np.zeros([len(self.b_obs_t), self.N])
                for itime in range(len(self.b_obs_t)):
                    self.d_b[itime] = obs_data[int(self.b_obs_t[itime] * (self.tstep-1)), self.N * 2: self.N * 3]
                    noise = np.random.randn(self.N)
                    noise = noise / np.linalg.norm(noise) * np.linalg.norm(self.d_b[itime]) * self.noise_level
                    self.d_b[itime] += noise

                true_param[irepeat] = self.params_true
                true_p0[irepeat] = self.p0_true

                self.p0 = np.zeros([4 * self.N])
                self.p0[self.N: 2 * self.N] = 1
                self.p0[3 * self.N: 4 * self.N] = 1

                self.params = np.zeros([7])
                self.params = np.exp(np.random.randn(7) * self.sigma + self.mu)

                inversion.inverse_p0_parameters(self)

                est_param[irepeat] = self.params
                est_p0[irepeat] = self.p0

                t = np.linspace(0, self.T, self.tstep)
                self.fitting_c = np.zeros_like(self.d_c)
                self.fitting_b = np.zeros_like(self.d_b)

                rhs = lambda t, s: forward_op.sys_rhs(s, t, self.L, list(self.params))
                est_s_seq = solve_ivp(rhs, [0, self.T], self.p0, t_eval=t, method='LSODA')['y'].T
                for itime in range(len(self.c_obs_t)):
                    self.fitting_c[itime] = est_s_seq[int(self.c_obs_t[itime] * (self.tstep - 1)), :self.N]
                for itime in range(len(self.b_obs_t)):
                    self.fitting_b[itime] = est_s_seq[int(self.b_obs_t[itime] * (self.tstep - 1)), self.N * 2: self.N * 3]

                rel_errs = np.linalg.norm(self.d_c.reshape([-1, ]) - self.fitting_c.reshape([-1, ])) / np.linalg.norm(self.d_c.reshape([-1, ]))
                r2_scores = r2_score(self.d_c.reshape([-1, ]), self.fitting_c.reshape([-1, ]))

                print(f'patient-wise r2: {r2_scores}, relative error: {rel_errs}')

            np.savez(self.save_folder + f'/25MedIA_syn_inversion_point_inversion_random_init_nobs{len(self.c_obs_t)}_noise{self.noise_level}_tauIC5_abIC10_no_atrophy_Template{self.template_id}.npz',
                     true_param=true_param, true_p0=true_p0,
                     c_obs_t = self.c_obs_t, b_obs_t=self.b_obs_t,
                     est_param=est_param, est_p0=est_p0
                     )

        return


    def eval_inversion_result(self):

        self.fitting_c = np.zeros_like(self.d_c)
        self.fitting_b = np.zeros_like(self.d_b)

        t = np.linspace(0, self.T, self.tstep)
        rhs = lambda t, s: forward_op.sys_rhs(s, t, self.L, list(self.params))
        est_s_seq = solve_ivp(rhs, [0, self.T], self.p0, t_eval=t, method='LSODA')['y'].T
        for itime in range(len(self.c_obs_t)):
            self.fitting_c[itime] = est_s_seq[int(self.c_obs_t[itime] / self.T * (self.tstep - 1)), :self.N]
        for itime in range(len(self.b_obs_t)):
            self.fitting_b[itime] = est_s_seq[int(self.b_obs_t[itime] / self.T * (self.tstep - 1)), self.N * 2: self.N * 3]

        fitting_result = np.concatenate([self.fitting_c, self.fitting_b], axis=0)
        obs_result = np.concatenate([self.d_c, self.d_b], axis=0)
        r2 = r2_score(obs_result.reshape([-1, ]), fitting_result.reshape([-1, ]))
        rel_err = np.linalg.norm(obs_result.reshape([-1, ]) - fitting_result.reshape([-1, ])) / np.linalg.norm(obs_result.reshape([-1, ]))
        print(f'tau r2: {r2_score(self.d_c.reshape([-1, ]), self.fitting_c.reshape([-1, ]))}')
        print(f'abeta r2: {r2_score(self.d_b.reshape([-1, ]), self.fitting_b.reshape([-1, ]))}')
        return r2, rel_err

    def run_inversion(self):

        if self.use_multiscan == 0:
            nrepeat = 1
        else:
            nrepeat = 10

        params_collect = []
        p0_collect = []
        r2_collect = []

        for irepeat in range(nrepeat):

            try:
                self.params = np.random.rand(7)
                if self.remove_param_loc != 10:
                    self.params[self.remove_param_loc] = 0
                self.p0 = np.zeros([4 * self.N])
                self.p0[self.N: 2 * self.N] = 1
                self.p0[3 * self.N: 4 * self.N] = 1

                start_t = time.time()
                inversion.inverse_p0_parameters(self)
                end_t = time.time()
                print('run time of inversion', end_t - start_t)

                r2, _ = self.eval_inversion_result()
                r2_collect.append(r2)
                params_collect.append(self.params)
                p0_collect.append(self.p0)

            except:
                continue

        best_loc = np.argmax(np.asarray(r2_collect))
        self.params = params_collect[best_loc]
        self.p0 = p0_collect[best_loc]
        r2, rel_err = self.eval_inversion_result()

        if self.use_multiscan == 0:
            str_nscan = 'single_scan'
        else:
            str_nscan = 'multi_scan'

        if self.save_result:
            np.savez(os.path.join(self.save_folder, f'inversion_sensitivity_random_init_pat_name_{self.pat_name}_{str_nscan}_{self.role}_tauIC{self.tau_sparsity}_abIC{self.abeta_sparsity}_no_atrophy_w_subcortical_Template{self.template_id}_ablation_{self.ablation_param}.npz'),
                     params_collect=np.asarray(params_collect), p0_collect=np.asarray(p0_collect), r2_collect=np.asarray(r2_collect))

        return r2, rel_err

    def clinical_inversion(self, ipat, tau_sparsity, abeta_sparsity):

        self.T = 1

        path_muse = self.project_path + '/data/muse/MUSE Template - Dictionary_ROI_Hierarchy.csv'
        dic_muse = pd.read_csv(path_muse)
        roi_list_all = dic_muse.loc[dic_muse['TISSUE_SEG'] == 'GM']['ROI_INDEX'].to_numpy()[5 + self.skip_nroi:]

        obs_tau_dict = {}
        obs_tau_dict['PID'] = []
        obs_abeta_dict = {}
        obs_abeta_dict['PID'] = []

        fit_tau_dict = {}
        fit_tau_dict['PID'] = []
        fit_abeta_dict = {}
        fit_abeta_dict['PID'] = []

        self.tau_sparsity = tau_sparsity
        self.abeta_sparsity = abeta_sparsity

        dps_file    = np.load(os.path.join(self.project_path, 'scripts/DPS_fitting.npz'))
        alpha_guess = dps_file['alpha_guess']
        beta_guess  = dps_file['beta_guess']
        patients    = dps_file['patient_name']

        adni_info_file = pd.read_csv(os.path.join(self.project_path, 'ADNI_combined_info_all_diag_Oct17_2024.csv'))
        tau_file = pd.read_csv(os.path.join(self.project_path, f'p_mmd_Template{self.template_id}_MUSE100.csv'))
        abeta_file = pd.read_csv(os.path.join(self.project_path, f'abeta_mmd_Template{self.template_id}_MUSE100.csv'))

        self.pat_name = patients[ipat]

        if self.use_multiscan == 0:
            str_nscan = 'single_scan'
        else:
            str_nscan = 'multi_scan'

        alpha = alpha_guess[ipat]
        beta = beta_guess[ipat]

        if self.use_multiscan == 1:
            tau_obs = np.asarray(tau_file.loc[tau_file['SubjectId'].str.contains(self.pat_name)].values[:, 6 + self.skip_nroi:]).astype('float')
            abeta_obs = np.asarray(abeta_file.loc[abeta_file['SubjectId'].str.contains(self.pat_name)].values[:, 6 + self.skip_nroi:]).astype('float')
        else:
            tau_obs = np.asarray(tau_file.loc[tau_file['SubjectId'].str.contains(self.pat_name)].values[-1:, 6 + self.skip_nroi:]).astype('float')
            abeta_obs = np.asarray(abeta_file.loc[abeta_file['SubjectId'].str.contains(self.pat_name)].values[-1:, 6 + self.skip_nroi:]).astype('float')

        adni_info_pat_file = adni_info_file.loc[adni_info_file['SubjectId']==self.pat_name]
        init_age = adni_info_pat_file['Initial_Age'].to_numpy()[0]
        mri_time = adni_info_pat_file['MRI_time'].to_numpy().reshape([-1, ])
        tau_pet_time = adni_info_pat_file['Tau-PET_time'].to_numpy().reshape([-1, ])
        abeta_pet_time = adni_info_pat_file['Abeta-PET_time'].to_numpy().reshape([-1, ])
        pat_init_time = mri_time[0].split(',')[0][2:12]

        c_obs_t = utils.extract_time_list(tau_pet_time[0], init_age, pat_init_time)
        self.c_obs_t = utils.dps_model(np.asarray(c_obs_t), alpha, beta)
        b_obs_t = utils.extract_time_list(abeta_pet_time[0], init_age, pat_init_time)
        self.b_obs_t = utils.dps_model(np.asarray(b_obs_t), alpha, beta)

        if self.use_multiscan == 0:
            self.c_obs_t = self.c_obs_t[-1:]
            self.b_obs_t = self.b_obs_t[-1:]

        self.d_c = tau_obs
        self.d_b = abeta_obs

        c_obs_len = np.minimum(len(self.c_obs_t), self.d_c.shape[0])
        self.c_obs_t = self.c_obs_t[:c_obs_len]
        self.d_c = self.d_c[:c_obs_len]

        b_obs_len = np.minimum(len(self.b_obs_t), self.d_b.shape[0])
        self.b_obs_t = self.b_obs_t[:b_obs_len]
        self.d_b = self.d_b[:b_obs_len]

        valid_scan_idx = self.comp_valid_scan_idx(self.c_obs_t)
        self.c_obs_t = self.c_obs_t[valid_scan_idx]
        self.d_c = self.d_c[valid_scan_idx]
        valid_scan_idx = self.comp_valid_scan_idx(self.b_obs_t)
        self.b_obs_t = self.b_obs_t[valid_scan_idx]
        self.d_b = self.d_b[valid_scan_idx]

        r2, rel_err = self.run_inversion()
        print('patient: {}, final result: relative error: {}; R2: {}'.format(self.pat_name, rel_err, r2))

        for i in range(self.d_c.shape[0]):
            obs_tau_dict['PID'].append(self.pat_name + '_{}'.format(i))
        for i in range(self.d_b.shape[0]):
            obs_abeta_dict['PID'].append(self.pat_name + '_{}'.format(i))

        for i in range(self.fitting_c.shape[0]):
            fit_tau_dict['PID'].append(self.pat_name + '_{}'.format(i))
        for i in range(self.fitting_b.shape[0]):
            fit_abeta_dict['PID'].append(self.pat_name + '_{}'.format(i))

        for (i_roi, roi) in enumerate(roi_list_all):
            obs_tau_dict['ROI {}'.format(roi)] = list(self.d_c[:, i_roi].reshape([-1, ]))
        for (i_roi, roi) in enumerate(roi_list_all):
            obs_abeta_dict['ROI {}'.format(roi)] = list(self.d_b[:, i_roi].reshape([-1, ]))

        for (i_roi, roi) in enumerate(roi_list_all):
            fit_tau_dict['ROI {}'.format(roi)] = list(self.fitting_c[:, i_roi].reshape([-1, ]))
        for (i_roi, roi) in enumerate(roi_list_all):
            fit_abeta_dict['ROI {}'.format(roi)] = list(self.fitting_b[:, i_roi].reshape([-1, ]))

        os.makedirs(self.save_folder, exist_ok=True)
        df_obs_tau = pd.DataFrame.from_dict(obs_tau_dict)
        df_obs_tau.to_csv(self.save_folder + f'/observation_tau_unnormalized_Template{self.template_id}_subj_{self.pat_name}_{self.role}_{str_nscan}_tauIC{self.tau_sparsity}_abIC{self.abeta_sparsity}_no_atrophy_w_subcortical_ablation_{self.ablation_param}.csv', index=False)
        df_obs_abeta = pd.DataFrame.from_dict(obs_abeta_dict)
        df_obs_abeta.to_csv(self.save_folder + f'/observation_abeta_unnormalized_Template{self.template_id}_subj_{self.pat_name}_{self.role}_{str_nscan}_tauIC{self.tau_sparsity}_abIC{self.abeta_sparsity}_no_atrophy_w_subcortical_ablation_{self.ablation_param}.csv', index=False)

        df_fit_tau = pd.DataFrame.from_dict(fit_tau_dict)
        df_fit_tau.to_csv(self.save_folder + f'/fitting_tau_unnormalized_Template{self.template_id}_subj_{self.pat_name}_{self.role}_{str_nscan}_tauIC{self.tau_sparsity}_abIC{self.abeta_sparsity}_no_atrophy_w_subcortical_ablation_{self.ablation_param}.csv', index=False)
        df_fit_abeta = pd.DataFrame.from_dict(fit_abeta_dict)
        df_fit_abeta.to_csv(self.save_folder + f'/fitting_abeta_unnormalized_Template{self.template_id}_subj_{self.pat_name}_{self.role}_{str_nscan}_tauIC{self.tau_sparsity}_abIC{self.abeta_sparsity}_no_atrophy_w_subcortical_ablation_{self.ablation_param}.csv', index=False)

    def clinical_multiscan_inversion_extrapolation(self, ipat, tau_sparsity, abeta_sparsity):

        self.T = 1

        path_muse = self.project_path + '/data/muse/MUSE Template - Dictionary_ROI_Hierarchy.csv'
        dic_muse = pd.read_csv(path_muse)
        roi_list_all = dic_muse.loc[dic_muse['TISSUE_SEG'] == 'GM']['ROI_INDEX'].to_numpy()[5:]

        obs_tau_dict = {}
        obs_tau_dict['PID'] = []
        obs_abeta_dict = {}
        obs_abeta_dict['PID'] = []

        extrap_tau_dict = {}
        extrap_tau_dict['PID'] = []
        extrap_abeta_dict = {}
        extrap_abeta_dict['PID'] = []

        self.tau_sparsity = tau_sparsity
        self.abeta_sparsity = abeta_sparsity

        dps_file    = np.load(os.path.join(self.project_path, 'scripts/DPS_fitting.npz'))
        alpha_guess = dps_file['alpha_guess']
        beta_guess  = dps_file['beta_guess']
        patients    = dps_file['patient_name']

        adni_info_file = pd.read_csv(os.path.join(self.project_path, 'ADNI_combined_info_all_diag_Oct17_2024.csv'))
        tau_file = pd.read_csv(os.path.join(self.project_path, f'p_mmd_Template{self.template_id}_MUSE100.csv'))
        abeta_file = pd.read_csv(os.path.join(self.project_path, f'abeta_mmd_Template{self.template_id}_MUSE100.csv'))

        self.pat_name = patients[ipat]

        if self.use_multiscan == 0:
            str_nscan = 'single_scan'
        else:
            str_nscan = 'multi_scan'


        alpha = alpha_guess[ipat]
        beta = beta_guess[ipat]

        tau_obs = np.asarray(tau_file.loc[tau_file['SubjectId'].str.contains(self.pat_name)].values[:, 6:]).astype('float')
        abeta_obs = np.asarray(abeta_file.loc[abeta_file['SubjectId'].str.contains(self.pat_name)].values[:, 6:]).astype('float')

        if len(tau_obs) < 2 or len(abeta_obs) < 2:
            return

        tau_target = tau_obs[-1, :]
        abeta_target = abeta_obs[-1, :]

        if self.use_multiscan == 1:
            tau_obs = tau_obs[:-1, :]
            abeta_obs = abeta_obs[:-1, :]
        if self.use_multiscan == 0:
            tau_obs = tau_obs[-2:-1, :]
            abeta_obs = abeta_obs[-2:-1, :]

        adni_info_pat_file = adni_info_file.loc[adni_info_file['SubjectId']==self.pat_name]
        init_age = adni_info_pat_file['Initial_Age'].to_numpy()[0]
        mri_time = adni_info_pat_file['MRI_time'].to_numpy().reshape([-1, ])
        tau_pet_time = adni_info_pat_file['Tau-PET_time'].to_numpy().reshape([-1, ])
        abeta_pet_time = adni_info_pat_file['Abeta-PET_time'].to_numpy().reshape([-1, ])
        pat_init_time = mri_time[0].split(',')[0][2:12]

        c_obs_t = utils.extract_time_list(tau_pet_time[0], init_age, pat_init_time)
        self.c_obs_t = utils.dps_model(np.asarray(c_obs_t), alpha, beta)
        b_obs_t = utils.extract_time_list(abeta_pet_time[0], init_age, pat_init_time)
        self.b_obs_t = utils.dps_model(np.asarray(b_obs_t), alpha, beta)

        if len(self.c_obs_t) < 2 and len(self.b_obs_t) < 2:
            return

        c_obs_t_target = self.c_obs_t[-1]
        b_obs_t_target = self.b_obs_t[-1]

        self.c_obs_t = self.c_obs_t[:-1]
        self.b_obs_t = self.b_obs_t[:-1]

        if self.use_multiscan == 0:
            self.c_obs_t = self.c_obs_t[-1:]
            self.b_obs_t = self.b_obs_t[-1:]

        self.d_c = tau_obs
        self.d_b = abeta_obs

        c_obs_len = np.minimum(len(self.c_obs_t), self.d_c.shape[0])
        self.c_obs_t = self.c_obs_t[:c_obs_len]
        self.d_c = self.d_c[:c_obs_len]

        b_obs_len = np.minimum(len(self.b_obs_t), self.d_b.shape[0])
        self.b_obs_t = self.b_obs_t[:b_obs_len]
        self.d_b = self.d_b[:b_obs_len]

        valid_scan_idx = self.comp_valid_scan_idx(self.c_obs_t)
        self.c_obs_t = self.c_obs_t[valid_scan_idx]
        self.d_c = self.d_c[valid_scan_idx]
        valid_scan_idx = self.comp_valid_scan_idx(self.b_obs_t)
        self.b_obs_t = self.b_obs_t[valid_scan_idx]
        self.d_b = self.d_b[valid_scan_idx]

        r2, rel_err = self.run_inversion()
        print('patient: {}, final result: relative error: {}; R2: {}'.format(self.pat_name, rel_err, r2))

        obs_tau_dict['PID'].append(self.pat_name)
        obs_abeta_dict['PID'].append(self.pat_name)

        extrap_tau_dict['PID'].append(self.pat_name)
        extrap_abeta_dict['PID'].append(self.pat_name)

        for (i_roi, roi) in enumerate(roi_list_all):
            obs_tau_dict['ROI {}'.format(roi)] = [tau_target[i_roi]]
        for (i_roi, roi) in enumerate(roi_list_all):
            obs_abeta_dict['ROI {}'.format(roi)] = [abeta_target[i_roi]]

        T = 1.5
        t = np.linspace(0, T, self.tstep)
        rhs = lambda t, s: forward_op.sys_rhs(s, t, self.L, self.params)
        s_seq = solve_ivp(rhs, [0, T], self.p0, t_eval=t, method='LSODA')['y'].T

        c_extrap = s_seq[int(c_obs_t_target / T * (self.tstep - 1)), :self.N]
        b_extrap = s_seq[int(b_obs_t_target / T * (self.tstep - 1)), self.N * 2: self.N * 3]

        print(f'extrapolation r2, tau: {r2_score(tau_target, c_extrap)}, abeta: {r2_score(abeta_target, b_extrap)}')

        for (i_roi, roi) in enumerate(roi_list_all):
            extrap_tau_dict['ROI {}'.format(roi)] = c_extrap[i_roi]
        for (i_roi, roi) in enumerate(roi_list_all):
            extrap_abeta_dict['ROI {}'.format(roi)] = b_extrap[i_roi]

        os.makedirs(self.save_folder, exist_ok=True)
        df_obs_tau = pd.DataFrame.from_dict(obs_tau_dict)
        df_obs_tau.to_csv(self.save_folder + f'/observation_tau_unnormalized_Template{self.template_id}_subj_{self.pat_name}_{self.role}_{str_nscan}_tauIC{self.tau_sparsity}_abIC{self.abeta_sparsity}_no_atrophy.csv', index=False)
        df_obs_abeta = pd.DataFrame.from_dict(obs_abeta_dict)
        df_obs_abeta.to_csv(self.save_folder + f'/observation_abeta_unnormalized_Template{self.template_id}_subj_{self.pat_name}_{self.role}_{str_nscan}_tauIC{self.tau_sparsity}_abIC{self.abeta_sparsity}_no_atrophy.csv', index=False)

        df_fit_tau = pd.DataFrame.from_dict(extrap_tau_dict)
        df_fit_tau.to_csv(self.save_folder + f'/extrap_tau_unnormalized_Template{self.template_id}_subj_{self.pat_name}_{self.role}_{str_nscan}_tauIC{self.tau_sparsity}_abIC{self.abeta_sparsity}_no_atrophy.csv', index=False)
        df_fit_abeta = pd.DataFrame.from_dict(extrap_abeta_dict)
        df_fit_abeta.to_csv(self.save_folder + f'/extrap_abeta_unnormalized_Template{self.template_id}_subj_{self.pat_name}_{self.role}_{str_nscan}_tauIC{self.tau_sparsity}_abIC{self.abeta_sparsity}_no_atrophy.csv', index=False)

    def cohort_inversion(self, tau_sparsity=5, abeta_sparsity=10):

        self.T = 1

        obs_tau_dict = {}
        obs_tau_dict['PID'] = []
        obs_abeta_dict = {}
        obs_abeta_dict['PID'] = []

        fit_tau_dict = {}
        fit_tau_dict['PID'] = []
        fit_abeta_dict = {}
        fit_abeta_dict['PID'] = []

        self.tau_sparsity = tau_sparsity
        self.abeta_sparsity = abeta_sparsity

        dps_file    = np.load(os.path.join(self.project_path, 'scripts/DPS_fitting.npz'))
        alpha_guess = dps_file['alpha_guess']
        beta_guess  = dps_file['beta_guess']
        patients    = dps_file['patient_name']

        adni_info_file = pd.read_csv(os.path.join(self.project_path, 'ADNI_combined_info_all_diag_Oct17_2024.csv'))
        tau_file = pd.read_csv(os.path.join(self.project_path, f'p_mmd_Template{self.template_id}_MUSE100.csv'))
        abeta_file = pd.read_csv(os.path.join(self.project_path, f'abeta_mmd_Template{self.template_id}_MUSE100.csv'))
        atrophy_file = pd.read_csv(os.path.join(self.project_path, f'vol_loss_Template4_MUSE100.csv'))

        time_gap = 0.05
        t_start_points = np.concatenate([np.arange(0, 1, time_gap), [1.05]])

        c_obs_t_all = defaultdict(list)
        d_c_all = defaultdict(list)

        b_obs_t_all = defaultdict(list)
        d_b_all = defaultdict(list)

        a_obs_t_all = defaultdict(list)
        d_a_all = defaultdict(list)

        for ipat in range(len(patients)):

            pat_name = patients[ipat]
            alpha = alpha_guess[ipat]
            beta = beta_guess[ipat]

            adni_info_pat_file = adni_info_file.loc[adni_info_file['SubjectId']==pat_name]
            init_age = adni_info_pat_file['Initial_Age'].to_numpy()[0]
            mri_time = adni_info_pat_file['MRI_time'].to_numpy().reshape([-1, ])
            tau_pet_time = adni_info_pat_file['Tau-PET_time'].to_numpy().reshape([-1, ])
            abeta_pet_time = adni_info_pat_file['Abeta-PET_time'].to_numpy().reshape([-1, ])
            pat_init_time = mri_time[0].split(',')[0][2:12]

            c_obs_t = utils.extract_time_list(tau_pet_time[0], init_age, pat_init_time)
            c_obs_t = utils.dps_model(np.asarray(c_obs_t), alpha, beta)
            b_obs_t = utils.extract_time_list(abeta_pet_time[0], init_age, pat_init_time)
            b_obs_t = utils.dps_model(np.asarray(b_obs_t), alpha, beta)
            a_obs_t = utils.extract_time_list(','.join(mri_time[0].split(',')[1:]), init_age, pat_init_time)
            a_obs_t = utils.dps_model(np.asarray(a_obs_t), alpha, beta)

            tau_obs = np.asarray(tau_file.loc[tau_file['SubjectId'].str.contains(pat_name)].values[:, 6:]).astype('float')
            abeta_obs = np.asarray(abeta_file.loc[abeta_file['SubjectId'].str.contains(pat_name)].values[:, 6:]).astype('float')
            atrophy_obs = np.asarray(atrophy_file.loc[atrophy_file['SubjectId'].str.contains(pat_name)].values[:, 6:]).astype('float')

            c_obs_len = np.minimum(len(c_obs_t), tau_obs.shape[0])
            c_obs_t = c_obs_t[:c_obs_len]
            d_c = tau_obs[:c_obs_len]
            for (i_ct, ct) in enumerate(c_obs_t):
                for i_interval in range(len(t_start_points) - 1):
                    if ct >= t_start_points[i_interval] and ct < t_start_points[i_interval + 1]:
                        c_obs_t_all[t_start_points[i_interval]].append(ct)
                        d_c_all[t_start_points[i_interval]].append(list(d_c[i_ct]))

            b_obs_len = np.minimum(len(b_obs_t), abeta_obs.shape[0])
            b_obs_t = b_obs_t[:b_obs_len]
            d_b = abeta_obs[:b_obs_len]
            for (i_bt, bt) in enumerate(b_obs_t):
                for i_interval in range(len(t_start_points) - 1):
                    if bt >= t_start_points[i_interval] and bt < t_start_points[i_interval + 1]:
                        b_obs_t_all[t_start_points[i_interval]].append(bt)
                        d_b_all[t_start_points[i_interval]].append(list(d_b[i_bt]))


            a_obs_len = np.minimum(len(a_obs_t), atrophy_obs.shape[0])
            a_obs_t = a_obs_t[:a_obs_len]
            d_a = atrophy_obs[:a_obs_len]
            for (i_at, at) in enumerate(a_obs_t):
                for i_interval in range(len(t_start_points) - 1):
                    if at >= t_start_points[i_interval] and at < t_start_points[i_interval + 1]:
                        a_obs_t_all[t_start_points[i_interval]].append(at)
                        d_a_all[t_start_points[i_interval]].append(list(d_a[i_at]))

        self.c_obs_t = []
        self.d_c = []
        self.d_c_std = []
        for key in sorted(list(c_obs_t_all.keys())):
            self.c_obs_t.append(np.asarray(c_obs_t_all[key]).mean())
            self.d_c.append(np.nanmean(np.asarray(d_c_all[key]), axis=0))
            self.d_c_std.append(np.nanstd(np.asarray(d_c_all[key]), axis=0))

        self.b_obs_t = []
        self.d_b = []
        self.d_b_std = []
        for key in sorted(list(b_obs_t_all.keys())):
            self.b_obs_t.append(np.asarray(b_obs_t_all[key]).mean())
            self.d_b.append(np.nanmean(np.asarray(d_b_all[key]), axis=0))
            self.d_b_std.append(np.nanstd(np.asarray(d_b_all[key]), axis=0))

        self.a_obs_t = []
        self.d_a = []
        self.d_a_std = []
        for key in sorted(list(a_obs_t_all.keys())):
            self.a_obs_t.append(np.asarray(a_obs_t_all[key]).mean())
            self.d_a.append(np.nanmean(np.asarray(d_a_all[key]), axis=0))
            self.d_a_std.append(np.nanstd(np.asarray(d_a_all[key]), axis=0))

        self.c_obs_t = np.asarray(self.c_obs_t)
        self.d_c = np.asarray(self.d_c)
        self.d_c_std = np.asarray(self.d_c_std)
        self.b_obs_t = np.asarray(self.b_obs_t)
        self.d_b = np.asarray(self.d_b)
        self.d_b_std = np.asarray(self.d_b_std)
        self.a_obs_t = np.asarray(self.a_obs_t)
        self.d_a = np.asarray(self.d_a)
        self.d_a_std = np.asarray(self.d_a_std)

        valid_scan_idx = self.comp_valid_scan_idx(self.c_obs_t)
        self.c_obs_t = self.c_obs_t[valid_scan_idx]
        self.d_c = self.d_c[valid_scan_idx]
        self.d_c_std = self.d_c_std[valid_scan_idx]
        valid_scan_idx = self.comp_valid_scan_idx(self.b_obs_t)
        self.b_obs_t = self.b_obs_t[valid_scan_idx]
        self.d_b = self.d_b[valid_scan_idx]
        self.d_b_std = self.d_b_std[valid_scan_idx]
        valid_scan_idx = self.comp_valid_scan_idx(self.a_obs_t)
        self.a_obs_t = self.a_obs_t[valid_scan_idx]
        self.d_a = self.d_a[valid_scan_idx]
        self.d_a_std = self.d_a_std[valid_scan_idx]

        np.savez('/Users/zwen/Downloads/cohort_obs.npz', \
                 obs_tau_mean=self.d_c, c_obs_t=self.c_obs_t, obs_abeta_mean=self.d_b, b_obs_t=self.b_obs_t, obs_atrophy_mean=self.d_a, a_obs_t=self.a_obs_t,
                 obs_tau_std=self.d_c_std, obs_abeta_std=self.d_b_std, obs_atrophy_std=self.d_a_std
                 )


    def comp_valid_scan_idx(self, obs_t):

        obs_t = np.unique(obs_t)
        threshold = self.delta_t * 2
        old_obs_t = np.asarray(obs_t).copy()
        obs_t = np.asarray(obs_t)

        def is_valid(obs_t, threshold):

            delta_obs_t = obs_t[1:] - obs_t[:-1]
            return not np.any(delta_obs_t < threshold)

        while not is_valid(obs_t, threshold):
            delta_obs_t = obs_t[1:] - obs_t[:-1]
            idx = np.where(delta_obs_t < threshold)[0][0]
            obs_t = np.hstack([obs_t[:idx], obs_t[idx + 1:]])

        remained_idx = []
        for idx in range(len(old_obs_t)):
            if old_obs_t[idx] in obs_t:
                remained_idx.append(idx)
        return remained_idx

if __name__ == '__main__':

    alz = Alzh_model_class('/Users/zwen/Desktop/codes/Alzh_tau_abeta_model_new')
    alz.forward_ablation_study()




