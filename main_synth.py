from scripts import pde_class
import argparse
import os

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Multi-biomarkers')
    parser.add_argument('--noise_level', default=0) ## example 0.1
    args = parser.parse_args()

    ## test synthetic sensitivity to initial guess

    project_path = '/home1/08171/zheyw1/Alzh_tau_abeta_model_new'
    alz = pde_class.Alzh_model_class(project_path)
    alz.noise_level = float(args.noise_level)
    alz.save_folder = '/scratch/08171/zheyw1/MedIA25_results/MedIA25_synth_inversion'

    os.makedirs(alz.save_folder, exist_ok=True)
    alz.synthetic_inversion()