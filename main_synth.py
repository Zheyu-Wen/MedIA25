from scripts import pde_class
import argparse
import os

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Multi-biomarkers')
    parser.add_argument('--noise_level', default=0) ## example 0.1
    parser.add_argument('--project_path', default='.')
    parser.add_argument('--save_folder', default='./results')
    args = parser.parse_args()

    ## test synthetic sensitivity to initial guess

    project_path = str(args.project_path)
    alz = pde_class.Alzh_model_class(project_path)
    alz.noise_level = float(args.noise_level)
    alz.save_folder = str(args.save_folder)

    os.makedirs(alz.save_folder, exist_ok=True)
    alz.synthetic_inversion()