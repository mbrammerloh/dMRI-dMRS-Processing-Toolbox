#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to fit dMRS data with models.

Developed by Malte Brammerloh and integrated into this pipeline by Rita Oliveira.  

Last changed Jan 2026
"""

import os
import pandas as pd
import math
import shutil
import numpy as np
import glob
from dmri_dmrs_toolbox.dmrs.dmrsmodel import  DMRSModel
from dmri_dmrs_toolbox.dmrs.dmrsdata import DMRSDataset
from dmri_dmrs_toolbox.misc.custom_functions import (create_directory,copy_files)
from dmri_dmrs_toolbox.misc.bids_structure import create_bids_structure

def Step2_fitting(cfg):
    
    subj_list       = cfg['subj_list'] 
    data_path       = cfg['data_path']     
    scan_list       = pd.read_excel(os.path.join(data_path, cfg['scan_list_name'] ))
    create_directory(os.path.join(cfg['data_path'],'derivatives',cfg['analysis_foldername']))

    ######## SUBJECT-WISE OPERATIONS ########
    for subj in  subj_list:
    
        print('Analysing dmrs of subject ' + subj + '...')
    
        # Extract data for subject
        subj_data = scan_list[
                            (scan_list['study_name'] == subj) &
                            (scan_list['acqType'] == 'SPECIAL') &
                            (scan_list['analyse'] == 'y')
                             ].reset_index(drop=True)
        
        if subj_data.empty:
            print(f"No data to analyse for subject {subj}, skipping.")
            continue
        
        # Generate paths 
        raw_path        = os.path.join(data_path, 'raw_data', list(subj_data['raw_data_folder'].unique())[0]) 

        # List of acquisition sessions
        sess_list    = [x for x in list(subj_data['sessNo'].unique()) if not math.isnan(x)] # clean NaNs
    
        ######## SESSION-WISE OPERATIONS ########
        for sess in sess_list :
            
            print(f"Analysing session {sess} ...")
            sess_data = subj_data[subj_data['sessNo'] == sess]
            
            bids_strc = create_bids_structure(subj=subj, sess=sess, datatype="dmrs", root=data_path, 
                                            folderlevel='derivatives', workingdir=cfg['prep_foldername'])
            
            # Loop for different Mixing Times (TM)
            TM_list             = np.unique(sess_data['TM'].astype(int).tolist())

            path_allTM      = os.path.join(bids_strc.get_path(),'allTM')
            if os.path.exists(path_allTM):
                shutil.rmtree(path_allTM)
            create_directory(path_allTM)

            ######## TM-WISE OPERATIONS ########
            for TM in TM_list:
                
                print(f"Copying data from mixing time: {TM} ms to common folder ...")

                path_quantified = os.path.join(bids_strc.get_path(),f'TM_{str(TM)}','quantified')
                
                for file in glob.glob(os.path.join(path_quantified, '*')):
                    destino = [os.path.join(path_allTM, os.path.basename(file))]
                    copy_files([file], destino)
                    #print(f'Copying {file} to \n {destino}...')

            # Initiate dataset
            dataset = DMRSDataset()
            dataset.load_lcm_quantified_directory(path_allTM, raw_path)
            print(
                "Built dataset with:\n"
                f"  diffusion times: {dataset.all_diffusion_times}\n"
                f"  b-values:        {dataset.all_b_values}"
            )

            ######## RUN MODEL FIT ########
            
            # Define metabolites and normalize signal
            dataset.metabolites= cfg['metabolites']
            dataset.normalize_signal()

            # Define paths
            bids_strc = create_bids_structure(subj=subj, sess=sess, datatype="dmrs", root=data_path,
                                            folderlevel='derivatives', workingdir=cfg['analysis_foldername'])
            path_modeling_results       = bids_strc.get_path()

            # Detect and remove b-decay outliers before fitting (optional)
            if cfg.get('outlier_detection', 0):
                dataset.detect_outliers(k_sigma=cfg.get('outlier_k_sigma', 3.0),
                                        rel_tol=cfg.get('outlier_rel_tol', 0.05))
                dataset.export_outliers(os.path.join(path_modeling_results, 'outliers.csv'))

            # Initiate model dmrs
            dmrsmodel = DMRSModel(dataset)

            # Loop through models
            for model in cfg['models']:
                
                print(f"Fitting model: {model}...")
                mc_draws=2

                this_spec = dmrsmodel.model_specs[model]
                if 'b_max' in cfg.keys():
                    if cfg['b_max']>0:
                        print("Applying b-value cutoff in all models of", cfg['b_max'], "ms/µm².")
                        if "b_max" in this_spec.options.keys():
                            this_spec.options["b_max"] = np.min([this_spec.options["b_max"],cfg['b_max']])
                        else:
                            this_spec.options["b_max"]=cfg['b_max']

                dmrsmodel.apply_model(
                    model,
                    modified_spec=this_spec
                )
            
                dmrsmodel.calculate_uncertainties(
                    model,
                    monte_carlo_draws=mc_draws,
                    modified_spec=this_spec)
                create_directory(os.path.join(path_modeling_results,model))
                dmrsmodel.plot_results(model, os.path.join(path_modeling_results,model),
                                       info=f"{subj} ses-{int(sess):02d}")
                dmrsmodel.print_results(model)
                dmrsmodel.export_csvs(model, os.path.join(path_modeling_results,model,"csvs"))
                