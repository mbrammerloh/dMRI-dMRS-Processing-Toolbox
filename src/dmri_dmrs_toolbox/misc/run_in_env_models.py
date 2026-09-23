#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  1 15:14:45 2025

@author: localadmin
"""
import sys
import numpy as np
import os

def get_parameter_limits(model_clean, extra):
    """Return parameter bounds for graymatter_swissknife models."""
    param_lims = None

    if model_clean == "Nexi":
        if extra == "ex_vivo":
            param_lims = np.array([
                [1, 150],
                [0.05, 2],
                [0.05, 2],
                [0.1, 0.9],
                [0, 100],
            ])
        elif extra == "in_vivo":
            param_lims = np.array([
                [1, 150],
                [0.1, 3.5],
                [0.1, 3.5],
                [0.1, 0.9],
                [0, 100],
            ])

    elif model_clean == "Sandix":
        if extra == "ex_vivo":
            param_lims = np.array([
                [1, 150],
                [0.1, 2],
                [0.1, 2],
                [0.05, 0.95],
                [1, 30],
                [0.05, 0.5],
                [0, 100],
            ])
        elif extra == "in_vivo":
            param_lims = np.array([
                [1, 150],
                [0.1, 3.5],
                [0.1, 3.5],
                [0.05, 0.95],
                [1, 30],
                [0.05, 0.5],
                [0, 100],
            ])
            
            # Experimental Sandix parameterization using angular fractions.
            # eps_sigma = 1e-6
            # eps_th = 1e-4                
            # f_n_min = 0.05; f_n_max = 0.95; f_s_max = 0.7; f_s_min = 0.05; 
            # theta1_lo = np.arccos(np.sqrt(f_n_max))  
            # theta1_hi = np.arccos(np.sqrt(f_n_min))  
            # theta2_lo = np.arccos(np.sqrt(f_s_max)) 
            # theta2_hi =  np.arccos(np.sqrt(f_s_min))  # doesn't really matter
            # param_lims = np.array([[1, 150], [0, 3], [0, 3], [eps_th, theta1_hi], [1, 15], [eps_th, theta2_hi], [eps_sigma, 100]])     
            
    elif model_clean == "Sandi":
        if extra == "ex_vivo":
            param_lims = np.array([
                [0.1, 2],
                [0.1, 2],
                [0.05, 0.95],
                [1, 30],
                [0.05, 0.5],
                [0, 100],
            ])
        elif extra == "in_vivo":
            param_lims = np.array([
                [0.1, 3.5],
                [0.1, 3.5],
                [0.05, 0.95],
                [1, 30],
                [0.02, 0.5],
                [0, 100],
            ])
            
            # Experimental Sandi parameterization using angular fractions.
            # f_n_min = 0.05; f_n_max = 0.95; f_s_max = 0.7; f_s_min = 0.05; 
            # theta1_lo = np.arccos(np.sqrt(f_n_max))  
            # theta1_hi = np.arccos(np.sqrt(f_n_min))  
            # theta2_lo = np.arccos(np.sqrt(f_s_max)) 
            # theta2_hi =  np.arccos(np.sqrt(f_s_min))  # doesn't really matter
            # param_lims = np.array([[0, 3], [0, 3], [theta1_lo, theta1_hi], [1, 15], [theta2_lo, theta2_hi], [0, 100]])     


    return param_lims


def Run_model():
   
    # Get arguments passed to the script
    model       = sys.argv[1] 
    has_ste     = ('wSTE' in model)
    has_mrs     = ('mrs_informed' in model) 
    model_clean  = model.split('_')[0] if (has_ste or has_mrs) else model 
 
    if model_clean == 'Nexi' or model_clean =='Sandi' or model_clean =='Smex' or model_clean =='Sandix':
       
        from graymatter_swissknife import estimate_model, estimate_model_noiseless
        out_path    = sys.argv[2]
        dwi_path    = sys.argv[3]  
        bvals_path  = sys.argv[4]
        Delta_path  = sys.argv[5]
        delta_path  = sys.argv[6]
        sigma_path  = sys.argv[7]
        mask_path   = sys.argv[8]
        extra       = sys.argv[9]
        algo_fit    = sys.argv[10]
        xgboost_model_path = sys.argv[11]
        debug       = '--debug' in sys.argv  # Set debug flag if passed

        # parameter limits  
        param_lims = get_parameter_limits(model_clean, extra)
        
        small_delta_val = float(np.loadtxt(delta_path)[0])
        est_kwargs = dict(
            model_name=model_clean,                
            dwi_path=dwi_path,
            bvals_path=bvals_path,
            delta_path=Delta_path,
            small_delta=small_delta_val,
            lowb_noisemap_path=sigma_path,
            out_path=out_path,
            mask_path=mask_path,
            adjust_parameter_limits=param_lims,
            debug=debug,
            save_nls_initialization=True,
            optimization_method = algo_fit, 
            xgboost_model_path = xgboost_model_path
            )
            
        # Only add uA_path if needed; 
        if has_ste:
            est_kwargs['STE_path'] = sys.argv[11]
            est_kwargs['bvals_path_ste'] = sys.argv[12]

        if has_mrs:
             est_kwargs['prior_rs_target'] = float(sys.argv[11])
             est_kwargs['prior_rs_sigma']  = float(sys.argv[12])
 
        # Estimate model
        estimate_model(**est_kwargs)
        
        
        # # OPTION NO NOISE
        # param_lims = param_lims[:-1]
        # est_kwargs = dict(
        #     model_name=model_clean,                
        #     dwi_path=dwi_path,
        #     bvals_path=bvals_path,
        #     delta_path=Delta_path,
        #     small_delta=small_delta_val,
        #     out_path=out_path,
        #     mask_path=mask_path,
        #     adjust_parameter_limits=param_lims,
        #     debug=debug,
        #     save_nls_initialization=True,
        #     optimization_method = algo_fit, 
        #     xgboost_model_path = xgboost_model_path
        #     )
        # estimate_model_noiseless(**est_kwargs)
            

    elif model =='SMI' or model=='SMI_wSTE':
         out_path    = sys.argv[2]
         dwi_path    = sys.argv[3]  
         mask_path   = sys.argv[4]
         sigma_path  = sys.argv[5]
         data_path   = sys.argv[6]
         others      = sys.argv[7]
         
         call = [f'docker run -v {data_path}:/data nyudiffusionmri/designer2:v2.0.13 tmi -SMI',
                f'{others}',
                f'-sigma {sigma_path}',
                f'-mask {mask_path}',
                f'{dwi_path}',
                f'{out_path}']
    
         print(' '.join(call))
         os.system(' '.join(call))
        
         call = [f'docker run -v {data_path}:/data nyudiffusionmri/designer2:v2.0.13 chmod -R 777 {out_path}']
         print(' '.join(call))
         os.system(' '.join(call))
         
         
    elif model=='Sandi_amico':
        out_path    = sys.argv[2]
        dwi_path    = sys.argv[3]  
        bvals_path  = sys.argv[4]
        Delta_path  = sys.argv[5]
        delta_path  = sys.argv[6]
        sigma_path  = sys.argv[7]
        mask_path   = sys.argv[8]
        extra       = sys.argv[9]
        bvecs_path  = sys.argv[10]
        TE          = float(sys.argv[11])
        debug       = '--debug' in sys.argv  # Set debug flag if passed


        # Prepare
        import amico
        os.chdir(out_path)
        amico.setup()
        ae = amico.Evaluation()
        ae.set_config('doDirectionalAverage',True)
        ae.set_config('doComputeRMSE',True)
        delta =   np.unique(np.loadtxt(Delta_path))[0]
        small_delta = np.unique(np.loadtxt(delta_path))[0]
        scheme_filename = os.path.join(out_path,'SANDI_scheme.txt')
        amico.util.sandi2scheme(bvals_path, bvecs_path, delta*1e-3, small_delta*1e-3, TE_data=TE*1e-3, schemeFilename=scheme_filename, bStep=100)
         
        # Load data
        ae.load_data(dwi_filename=dwi_path, scheme_filename=scheme_filename, mask_filename=mask_path, b0_thr=1)
         
        # Compute response functions
        ae.set_model('SANDI')
        d_is = 3e-3 # Intra-soma diffusivity [mm^2/s]
        Rs = np.array([1.55555556e-06, 3.44444444e-06, 4.44444444e-06, 5.33333333e-06, 6.00000000e-06, 6.55555556e-06, 8.11111111e-06, 9.55555556e-06, 1.16666667e-05]) # Radii of the soma [meters]
        d_in = np.array([0.00091667, 0.00169444, 0.003]) # Intra-neurite diffusivitie(s) [mm^2/s]
        d_isos = np.array([0.00036111, 0.00163889, 0.003]) # Extra-cellular isotropic mean diffusivitie(s) [mm^2/s]
        ae.model.set(d_is, Rs, d_in, d_isos)
        ae.generate_kernels(regenerate=True, ndirs=1)
        ae.load_kernels()
        
        # Fit the model
        lambda1 = 0 # L1 regularization term (MUST be varied according to data)
        lambda2 = 5e-3 # L2 regularization term (MUST be varied according to data)
        ae.set_solver(lambda1=lambda1, lambda2=lambda2)
        ae.fit()
        ae.save_results(save_dir_avg=True)
    
    
if __name__ == "__main__":
    Run_model()
    