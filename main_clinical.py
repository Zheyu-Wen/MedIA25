from scripts import pde_class
import argparse
import os
import numpy as np
import pandas as pd

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Multi-biomarkers')
    parser.add_argument('--pat_ids', default=0) ## example 0
    parser.add_argument('--tau_sparsity', default=5)
    parser.add_argument('--abeta_sparsity', default=10)
    parser.add_argument('--role', default='fitting')
    parser.add_argument('--use_multiscan', default=0)
    parser.add_argument('--ablation_param', default='none')
    parser.add_argument('--project_path', default='.')
    parser.add_argument('--save_folder', default='./results')


    args = parser.parse_args()

    pat_id = int(args.pat_ids)
    tau_sparsity = int(args.tau_sparsity)
    abeta_sparsity = int(args.abeta_sparsity)
    use_multiscan = int(args.use_multiscan)

    project_path = str(args.project_path)
    alz = pde_class.Alzh_model_class(project_path)
    alz.save_folder = str(args.save_folder)

    alz.role = args.role
    alz.use_multiscan = use_multiscan
    if args.ablation_param == 'kappa_c':
        alz.kappa_c_decay = 0
    if args.ablation_param == 'rho_c':
        alz.rho_c_decay = 0
    if args.ablation_param == 'gamma_c':
        alz.gamma_c_decay = 0
    if args.ablation_param == 'rho_cb':
        alz.rho_cb_decay = 0
    if args.ablation_param == 'kappa_b':
        alz.kappa_b_decay = 0
    if args.ablation_param == 'rho_b':
        alz.rho_b_decay = 0
    if args.ablation_param == 'gamma_b':
        alz.gamma_b_decay = 0

    if args.ablation_param != 'none':
        alz.remove_param_loc = ['kappa_c', 'rho_c', 'gamma_c', 'rho_cb', 'kappa_b', 'rho_b', 'gamma_b'].index(args.ablation_param)
    else:
        alz.remove_param_loc = 10

    alz.ablation_param = args.ablation_param
    os.makedirs(alz.save_folder, exist_ok=True)
    if args.role == 'fitting':
        alz.clinical_inversion(pat_id, tau_sparsity, abeta_sparsity)
    elif args.role == 'extrap':
        alz.clinical_multiscan_inversion_extrapolation(pat_id, tau_sparsity, abeta_sparsity)