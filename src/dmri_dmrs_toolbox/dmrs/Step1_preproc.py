#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to preprocess dMRS data.

It uses matlab codes provided by EPFL group of Cristina Cudalbu 
(see https://www.epfl.ch/labs/mrs4brain/ressources/mrs4brain-toolbox/)
and curated by Malte Brammerloh to:
    
- Convert bruker data to mat file
- Process dMRS data
- Quantify metabolites with LC model

Integrated into this pipeline by Rita Oliveira.  
Last changed Jan 2026
"""

import os
import pandas as pd
import math
import numpy as np
import nibabel as nib
import shutil
import glob
import re
import subprocess
from importlib.resources import files
from dmri_dmrs_toolbox.misc.bids_structure import create_bids_structure
from dmri_dmrs_toolbox.misc.custom_functions import create_directory

def Step1_preproc(cfg):
    
    
    subj_list = cfg['subj_list'] 
    data_path = cfg['data_path']     
    scan_list = pd.read_excel(os.path.join(data_path, cfg['scan_list_name'] ))
    
    ######## SUBJECT-WISE OPERATIONS ########
    for subj in  subj_list:
    
        print('Processing dmrs of subject ' + subj + '...')
    
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
            
            print(f"Processing session {sess} ...")
            sess_data = subj_data[subj_data['sessNo'] == sess]
            
            bids_strc = create_bids_structure(subj=subj, sess=sess, datatype="dmrs", root=data_path, 
                                            folderlevel='derivatives', workingdir=cfg['prep_foldername'])
            
            # Loop for different Mixing Times (TM)
            TM_list             = np.unique(sess_data['TM'].astype(int).tolist())

            ######## TM-WISE OPERATIONS ########
            for TM in TM_list:
                            
                print(f"Processing mixing time: {TM} ms ...")
                
                # Get the scan numbers for the water reference. Assumes there is only one
                subj_data_water = subj_data[
                                    (subj_data['sessNo'] == sess) &
                                    (subj_data['dMRS_acq_type'] == 'water') &
                                    (subj_data['TM'] == TM) 
                                     ].reset_index(drop=True)
                #water_reference_sequence_number =  subj_data_water['scanNo'].iloc[0]
                
                # Get the scan numbers for the metabolite data 
                subj_data_metab = subj_data[
                                    (subj_data['sessNo'] == sess) &
                                    (subj_data['dMRS_acq_type'] == 'metab') &
                                    (subj_data['TM'] == TM)
                                     ].reset_index(drop=True)
                metab_sequence_number = subj_data_metab['scanNo'].dropna().str.split(',').explode().astype(int).tolist()
    
                # Define paths
                input_path        = raw_path
                output_path       = os.path.join(bids_strc.get_path(),f'TM_{str(TM)}')
                
                # Delete output folder if redo_processing is on
                if cfg['redo_processing']==1 and os.path.exists(output_path):
                        print("Your previous results will be deleted and will be processed again")
                        shutil.rmtree(output_path)
                
                # Do analysis if it wasn't done before or if redo_processing is on
                if cfg['redo_processing'] == 1 or not os.path.exists(output_path):
                    
                    create_directory(output_path)
                
                    scan_list_format_matlab = "[" + " ".join(map(str, metab_sequence_number)) + "]"
                    coil_type         = subj_data_metab['coil_type'].iloc[0]
                    toolbox_path      = files("dmri_dmrs_toolbox.dmrs").joinpath("dSPECIAL_matlab_codes_Toi")
                    LCMpath           = cfg['LC_model_path']
        
                    # Chose basis set with TM closer to the acquired TM
                    basis_sets_folder = cfg['basis_set']
                    basis_list        = glob.glob(os.path.join(basis_sets_folder, "*.BASIS"))
                    tm_candidates = []
                    for basis_file in basis_list:
                        match = re.search(r"TM(\d+)", basis_file)
                        if match:
                            tm_value = int(match.group(1))
                            tm_candidates.append((tm_value, basis_file))
                    tm_closest, basis_set = min(tm_candidates,
                        key=lambda x: abs(x[0] - TM))
                    
                    
                    # Sh command of matlab files
                    cmd = [f"{toolbox_path}/run_processing_dmrs_matlab.sh",
                           cfg['MATLAB_Runtime'],
                        input_path,
                        output_path,
                        scan_list_format_matlab,
                        coil_type,
                        basis_set,
                        LCMpath,
                        "1"]
                    
                    print("\nShell command:")
                    print(" ".join(cmd))
                    print()
                    result = subprocess.run(cmd, check=False)
                    
                
                # Save in nifti file
                from dmri_dmrs_toolbox.dmrs.dmrsdata import DMRSDataset

                # Build dataset
                path_quantified = os.path.join(bids_strc.get_path(),f'TM_{str(TM)}','quantified')
                path_nifti = os.path.join(bids_strc.get_path(),f'TM_{str(TM)}','nifti_quantified')
                create_directory(path_nifti)
                dataset = DMRSDataset()
                dataset.load_lcm_quantified_directory(path_quantified, raw_path)
                print(
                    "Built dataset with:\n"
                    f"  diffusion times: {dataset.all_diffusion_times}\n"
                    f"  b-values:        {dataset.all_b_values}"
                )
               
                # Loop through metabolities and save each metabolite as separate nifti
                for metab in dataset.metabolites:
                    
                    signal = np.asarray(dataset.signal[metab], dtype=np.float32).squeeze()
                    bvals = np.asarray(dataset.b_values).squeeze()
                    
                    n_volumes = len(bvals)
                    
                    if signal.size != n_volumes:
                        raise ValueError(
                            f"Signal contains {signal.size} values, but bvals contains "
                            f"{n_volumes} volumes."
                        )
                    
                    # Shape: 1 voxel × number of diffusion volumes
                    data = signal.reshape(1, 1, 1, n_volumes)
                    
                    img = nib.Nifti1Image(data, affine=np.eye(4))
                    nib.save(img, os.path.join(path_nifti,f'dmrs_{metab}.nii.gz'))


                  
                  