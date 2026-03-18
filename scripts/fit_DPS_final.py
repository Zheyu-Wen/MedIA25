import numpy as np
import pandas as pd
from scripts import DPS_utils
import scipy.optimize as sci_opti
import warnings
from scipy.stats import norm
warnings.filterwarnings("ignore")

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

def assign_biomarkers(M, ages, i_pat, i_metric, data, data_time, init_age, init_time):

    pat_data_time = []
    if ',' in data_time:
        data_time_split = data_time.split(',')
        for (i_t, time) in enumerate(data_time_split):
            t = data_time_split[i_t]
            if '\'' in t:
                pat_time = t[2:-2]
            else:
                pat_time = t[1:-1]
            if pat_time == 'nan':
                continue
            pat_data_time.append(comp_delta_year(init_time, pat_time) + init_age)
        for j in range(np.minimum(len(data), np.minimum(M.shape[2], len(pat_data_time)))):
            ages[i_metric, i_pat, j] = pat_data_time[j]
            M[i_metric, i_pat, j] = data[j]

    else:
        if '\'' in data_time:
            pat_time = data_time[2:-2]
        else:
            pat_time = data_time[1:-1]
        ages[i_metric, i_pat, 0] = float(comp_delta_year(init_time, pat_time) + init_age)
        M[i_metric, i_pat, 0] = data[0]

    return M, ages

adni_info_file = pd.read_csv('../ADNI_combined_info_all_diag_Oct17_2024.csv')
pat_list = adni_info_file['SubjectId'].to_numpy().reshape([-1, ])

n_pat = pat_list.shape[0]
# n_mental_score = 3
n_mental_score = 2
n_sample = 3

ages = np.zeros([n_mental_score, n_pat, n_sample])
M = np.zeros([n_mental_score, n_pat, n_sample])

init_age = adni_info_file['Initial_Age'].to_numpy().reshape([-1, ])
mri_time = adni_info_file['MRI_time'].to_numpy().reshape([-1, ])
tau_time = adni_info_file['Tau-PET_time'].to_numpy().reshape([-1, ])
abeta_time = adni_info_file['Abeta-PET_time'].to_numpy().reshape([-1, ])

adas_score = adni_info_file['ADAS13'].to_numpy().reshape([-1, ])
adas_time = adni_info_file['ADAS13_time'].to_numpy().reshape([-1, ])
mmse_score = adni_info_file['MMSE'].to_numpy().reshape([-1, ])
mmse_time = adni_info_file['MMSE_time'].to_numpy().reshape([-1, ])
ef_score = adni_info_file['ADNI_EF'].to_numpy().reshape([-1, ])
mem_score = adni_info_file['ADNI_Mem'].to_numpy().reshape([-1, ])
efmem_time = adni_info_file['ADNI_EF_Mem_time'].to_numpy().reshape([-1, ])
csf_time = adni_info_file['CSF_biomarker_time'].to_numpy().reshape([-1, ])
csfbeta_score = adni_info_file['CSF_Abeta42'].to_numpy().reshape([-1, ])
csftau_score = adni_info_file['CSF_pTau'].to_numpy().reshape([-1, ])
groups = adni_info_file['Group'].to_numpy().reshape([-1, ])

abeta_file = pd.read_csv('../abeta_mmd_Template4_MUSE100.csv')
tau_file = pd.read_csv('../p_mmd_Template4_MUSE100.csv')
atrophy_file = pd.read_csv('../vol_loss_Template4_MUSE100.csv')

valid_pat = 0
patient_list = []
group_list = []
for (i_pat, pat) in enumerate(pat_list):

    pat_init_age = init_age[i_pat]
    pat_init_time = mri_time[i_pat].split(',')[0][2:12]
    if len(mri_time[i_pat].split(',')) > 1:
        pat_atrophy_time = ','.join(mri_time[i_pat].split(',')[1:])
    else:
        pat_atrophy_time = mri_time[i_pat]
    pat_tau_time = tau_time[i_pat]
    pat_abeta_time = abeta_time[i_pat]

    pat_tau_data = np.asarray(tau_file.loc[tau_file['SubjectId'].str.contains(pat)].values[:, 6:]).astype('float')
    pat_abeta_data = np.asarray(abeta_file.loc[abeta_file['SubjectId'].str.contains(pat)].values[:, 6:]).astype('float')
    pat_atrophy_data = np.asarray(atrophy_file.loc[atrophy_file['SubjectId'].str.contains(pat)].values[:, 6:]).astype('float')

    # if len(pat_tau_data) < 1 or len(pat_abeta_data) < 1 or len(pat_atrophy_data) < 1:
    #     continue
    if len(pat_tau_data) < 1 or len(pat_abeta_data) < 1:
        continue
    M, ages = assign_biomarkers(M, ages, valid_pat, 0, pat_tau_data.sum(axis=1).reshape([-1, ]), pat_tau_time, pat_init_age, pat_init_time)
    M, ages = assign_biomarkers(M, ages, valid_pat, 1, pat_abeta_data.sum(axis=1).reshape([-1, ]), pat_abeta_time, pat_init_age, pat_init_time)
    # M, ages = assign_biomarkers(M, ages, valid_pat, 2, pat_atrophy_data.sum(axis=1).reshape([-1, ]), pat_atrophy_time, pat_init_age, pat_init_time)

    patient_list.append(pat)
    group_list.append(groups[i_pat])
    valid_pat += 1

M = M[:, :valid_pat]
ages = ages[:, :valid_pat]

ages[np.where(np.isnan(M))] = 0
M[np.where(np.isnan(M))] = 0

age_mask = np.zeros_like(ages)
age_mask[ages > 0] = 1

n_mental_score = M.shape[0]

M *= age_mask
for i in range(n_mental_score):

    mental_scores = M[i]
    mental_scores = mental_scores[mental_scores > 0]
    mental_min = mental_scores.min()
    mental_range = mental_scores.max() - mental_min
    M[i] = (M[i] - mental_min) / mental_range

M *= age_mask

n_pat = M.shape[1]
age_mask = np.zeros_like(ages)
age_mask[ages > 0] = 1
age_mask[M==0] = 0

nrepeat = 10
min_obj_val = 1
for repeat in range(nrepeat):
    alpha_guess = np.zeros([n_pat, ])
    beta_guess = np.zeros([n_pat, ])
    vec_theta_guess = np.random.rand(n_mental_score * 3)

    objective_func = lambda params: DPS_utils.obj_DPS(M, ages, params)
    params = np.concatenate((alpha_guess, beta_guess, vec_theta_guess), axis=0)
    bounds = []
    for i in range(n_pat):
        bounds.append(tuple([0, np.infty]))
    for i in range(n_pat):
        bounds.append(tuple([-np.infty, np.infty]))
    for i in range(3*n_mental_score):
        if i % 3 == 1:
            bounds.append(tuple([0, np.infty]))
        else:
            bounds.append(tuple([-np.infty, np.infty]))

    params_est, fmin, Dict = sci_opti.fmin_l_bfgs_b(objective_func, params, bounds=bounds)

    alpha_guess, beta_guess, vec_theta_guess = params_est[:n_pat], params_est[n_pat: 2*n_pat], params_est[2*n_pat:]
    theta_guess = np.zeros([n_mental_score, 3])
    for i in range(n_mental_score):
        theta_guess[i, 0] = vec_theta_guess[i * 3]
        theta_guess[i, 1] = vec_theta_guess[i * 3 + 1]
        theta_guess[i, 2] = vec_theta_guess[i * 3 + 2]

    Mhat, tau_hat = DPS_utils.mental_score_model(ages, alpha_guess, beta_guess, theta_guess)
    Merr = np.linalg.norm(M - Mhat * age_mask) / np.linalg.norm(M)
    if Merr < min_obj_val:
        print(Merr)
        min_obj_val = Merr
        np.savez('DPS_fitting_tau_abeta_modal.npz', alpha_guess=alpha_guess, beta_guess=beta_guess,
                                    theta_guess=theta_guess, DPS=tau_hat, Merr=Merr,
                                    ages=ages, M=M, patient_name=np.asarray(patient_list), group=np.asarray(group_list), mask=age_mask)

