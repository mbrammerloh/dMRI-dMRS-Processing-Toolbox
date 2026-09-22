"""
Script to convert preprocess dMRI data.
Includes: denoising, gibbs correction, top up, eddy
It can be run for the combined dataset (all diffusion times) or 
for one diffusion time only (the longest, for WM fitting).
It does not use a particular python environment (base).

Last changed Jan 2025
@author: Rita O
"""

import os
import pandas as pd
import math
import numpy as np
from dmri_dmrs_toolbox.misc.bids_structure import create_bids_structure
from dmri_dmrs_toolbox.misc.custom_functions import *
import matplotlib.pyplot as plt
from importlib.resources import files
import shutil
plt.close('all')

def Step3_preproc_STE(cfg):
    
    data_path   = cfg['data_path']     
    scan_list   = pd.read_excel(os.path.join(data_path, cfg['scan_list_name']))
    topupon     = cfg['do_topup'] 
    
    # update cfg in case one of the steps was 1 and the next ones 0, it changes the subsequents to 1 as well
    update_cfg(cfg)
    
    ######## SUBJECT-WISE OPERATIONS ########
    for subj in cfg['subj_list']:
    
        print('Preprocessing ' + subj + '...')
        
        # Extract data for subject
        subj_data      = scan_list[(scan_list['study_name'] == subj)].reset_index(drop=True)
        subj_data      = subj_data[subj_data['analyse'] == 'y'].reset_index(drop=True)

        # List of available acquisition sessions
        all_sessions = sorted(
            int(x) for x in subj_data["sessNo"].unique() if not math.isnan(x)
        )
        
        # Use all sessions unless specific ones are requested
        if cfg["sess_list"] is None:
            sess_list = all_sessions
        else:
            sess_list = [s for s in cfg["sess_list"] if s in all_sessions]
        
        # Check that data exists
        if not np.any(np.array(subj_data['acqType']) == 'STE'):
                print("No dwi scans with STE found — exiting.")
                continue  
              
        ######## SESSION-WISE OPERATIONS ########
        for sess in sess_list:
            
            subj_data_sess = subj_data[subj_data["sessNo"] == sess]
            subj_data_sess = subj_data_sess.reset_index(drop=True)
          
            print('Working on session ' + str(sess) + '...')
            
            # Copy nifti data to preprocessing folder
            nifti_path      = os.path.join(data_path, 'nifti_data', 'sorted', subj,f"ses-{sess:02}",'dwi_STE')
            preproc_path    = os.path.join(data_path, 'derivatives', cfg['prep_foldername'], subj,f"ses-{sess:02}",'dwi_STE')
            if not os.path.exists(preproc_path) or cfg['redo_all']:
                if os.path.exists(preproc_path):
                    print("Your previous results will be deleted, are you sure? Press any key to continue or abort.")
                    input()
                    shutil.rmtree(preproc_path)
                
                shutil.copytree(nifti_path, preproc_path)

            # Define anat bids structure
            bids_strc_anat = create_bids_structure(subj=subj, sess=sess, datatype="anat", root=data_path, 
                                        folderlevel='derivatives', workingdir=cfg['prep_foldername'])
            
            ########################## DWI PROCESSING PREPARATION ##########################
            bids_strc = create_bids_structure(subj=subj, sess=sess, datatype="dwi_STE", root=data_path, 
                                         folderlevel='derivatives', workingdir=cfg['prep_foldername'])
            # Index of diff scans for this session 
            dwi_indices = np.where(
                (np.array(subj_data_sess['acqType']) == 'STE') &
                (np.array(subj_data_sess['sessNo']) == sess))[0]

            # Generate paths for fwd and rev acquisition types
            masks_paths = []; paths_to_process = []; paths_b0_fwd =[];  paths_dwi_fwd = []; paths_b0_rev =[]; paths_dwi_rev =[]; 
            for scn_ctr in dwi_indices:    
                bids_strc.set_param(description='STE_' + subj_data_sess['phaseDir'][scn_ctr])
                if subj_data_sess['phaseDir'][scn_ctr] == 'fwd' :
                    paths_dwi_fwd.append(bids_strc.get_path('dwi.nii.gz'))
                    paths_b0_fwd.append(bids_strc.get_path('b0.nii.gz'))
                elif subj_data_sess['phaseDir'][scn_ctr] == 'rev':
                    paths_dwi_rev.append(bids_strc.get_path('dwi.nii.gz'))
                    paths_b0_rev.append(bids_strc.get_path('b0.nii.gz'))
                                  
            # B0 extract for each scan and REGISTRATION of brain mask with avg B0 - fwd scans
            for paths_dwi in paths_dwi_fwd:
                if paths_dwi and not os.path.exists(paths_dwi.replace('dwi.nii.gz', 'b0_mask.nii.gz')):
                    
                    # Extract b0 volume
                    extract_vols(paths_dwi, paths_dwi.replace('dwi.nii.gz', 'b0.nii.gz'), 0, 1, cfg)
                    
                    # Bias field correct b0
                    N4_unbias(paths_dwi.replace('dwi.nii.gz', 'b0.nii.gz'), paths_dwi.replace('dwi.nii.gz', 'b0_bc.nii.gz'),cfg)
            
                    # Register dwi --> T2w
                    antsreg_simple(bids_strc_anat.get_path('T2w_bc.nii.gz'),  # fixed
                            paths_dwi.replace('dwi.nii.gz', 'b0_bc.nii.gz'),  # moving
                            paths_dwi.replace('dwi.nii.gz', 'dwi2T2w'),cfg)
            
                    # Apply inverse transform to put T2w in dwi space
                    ants_apply_transforms_simple([bids_strc_anat.get_path('T2w_bc.nii.gz'),
                                           bids_strc_anat.get_path('T2w_bc_brain.nii.gz')],  # input 
                                        paths_dwi.replace('dwi.nii.gz', 'b0_bc.nii.gz'),  # moving
                                        [paths_dwi.replace('dwi.nii.gz', 'T2w_in_dwi.nii.gz'),
                                         paths_dwi.replace('dwi.nii.gz', 'T2w_brain_in_dwi.nii.gz')],  # output
                                        [paths_dwi.replace('dwi.nii.gz', 'dwi2T2w0GenericAffine.mat'), 1], cfg)  # transform 1
                                        #paths_dwi.replace('dwi.nii.gz', 'dwi2T2w1InverseWarp.nii.gz'),cfg)  # transform 2
            
                    # QA
                    QA_reg(paths_dwi.replace('dwi.nii.gz', 'T2w_brain_in_dwi.nii.gz'),
                           paths_dwi.replace('dwi.nii.gz', 'b0_bc.nii.gz'),
                           os.path.join(os.path.dirname(paths_dwi), 'QA_reg'),cfg)
            
                    # Create mask
                    make_mask(paths_dwi.replace('dwi.nii.gz', 'T2w_brain_in_dwi.nii.gz'),
                              paths_dwi.replace('dwi.nii.gz', 'b0_mask.nii.gz'), 100, cfg)
                    
                # save masks path to be used later
                masks_paths.append(paths_dwi.replace('dwi.nii.gz', 'b0_mask.nii.gz'))
           
            # B0 extract for each scan - rev scans
            for paths_dwi in paths_dwi_rev:
                if paths_dwi and not os.path.exists(paths_dwi.replace('dwi.nii.gz', 'b0_mask.nii.gz')):
                    
                    # Extract b0 volume
                    extract_vols(paths_dwi, paths_dwi.replace('dwi.nii.gz', 'b0.nii.gz'), 0, 1, cfg)
            
            # Copy files to working folder
            bids_strc.set_param(description='STE_fwd')
            if paths_dwi_fwd:
                concat_niftis(paths_b0_fwd, bids_strc.get_path('b0_fwd.nii.gz'), 1)
            if paths_dwi_rev:
                 concat_niftis(paths_b0_rev, bids_strc.get_path('b0_rev.nii.gz'), 1) # assumes only one B0 value was collected in rev direction 
                    
            ########################## DWI PROCESSING ##########################       
            if paths_dwi_fwd: # only analyse if there is that type of data
                
                # Set output path
                bids_strc.set_param(description='STE_fwd')
                create_directory(bids_strc.get_path())
                output_path = bids_strc.get_path();
         
                # Create deformed mask
                union_niftis(masks_paths, bids_strc.get_path('mask_before_preproc.nii.gz'))
                #filter_clusters_by_size(bids_strc.get_path('mask.nii.gz'), bids_strc.get_path('mask.nii.gz'), 200)
    
                # DENOISE
                if not os.path.exists(bids_strc.get_path('dwi_dn.nii.gz')) or cfg['redo_denoise']:
                    script_path = files("dmri_dmrs_toolbox.dwi")
                    if cfg['algo_denoising']=='matlab_MPPCA' or cfg['algo_denoising']=='mrtrix_MPPCA' or cfg['algo_denoising']=='matlab_tMPPCA_4D':
                      #  denoise_matlab(bids_strc.get_path('dwi.nii.gz'), bids_strc.get_path('dwi_dn.nii.gz'), bids_strc.get_path('DiffTime.txt'), script_path, cfg,'MPPCA')
                    #elif cfg['algo_denoising']=='mrtrix_MPPCA':
                        denoise_vols_default_kernel(bids_strc.get_path('dwi.nii.gz'), bids_strc.get_path('dwi_dn.nii.gz'), bids_strc.get_path('dwi_dn_sigma.nii.gz'),cfg)
                    # elif cfg['algo_denoising']=='matlab_tMPPCA_4D':
                    #     denoise_matlab(bids_strc.get_path('dwi.nii.gz'), bids_strc.get_path('dwi_dn.nii.gz'), bids_strc.get_path('DiffTime.txt'), script_path, cfg,'tMPPCA-4D')
                    elif cfg['algo_denoising']=='designer_tMPPCA' or cfg['algo_denoising']=='matlab_tMPPCA_5D':
                         denoise_designer(bids_strc.get_path('dwi.nii.gz'), bids_strc.get_path('bvecs_fake.txt'), bids_strc.get_path('bvalsNom.txt'), bids_strc.get_path('dwi_dn.nii.gz'), data_path, 'jespersen', cfg)

                    calc_snr(bids_strc.get_path('dwi.nii.gz'), bids_strc.get_path('dwi_dn_sigma.nii.gz'),bids_strc.get_path('dwi_snr.nii.gz'), cfg)
                    QA_denoise(bids_strc, 'dwi_dn_res.nii.gz','dwi_dn_sigma.nii.gz',os.path.join(output_path, 'QA_denoise'), cfg)

                # GIBBS UNRINGING
                if not os.path.exists(bids_strc.get_path('dwi_dn_gc.nii.gz')) or cfg['redo_gibbs']:
                    gibbs_corr(bids_strc.get_path('dwi_dn.nii.gz'), bids_strc.get_path('dwi_dn_gc.nii.gz'),cfg)
                    QA_gc(bids_strc, 'dwi_dn.nii.gz', 'dwi_dn_gc.nii.gz', os.path.join(output_path, 'QA_gc'), cfg)

                # TOPUP
                if (not os.path.exists(bids_strc.get_path('dwi_dn_gc_topup.nii.gz')) or cfg['redo_topup']) and topupon and paths_dwi_rev:
                    topup_routine(bids_strc.get_path('dwi_dn_gc.nii.gz'), bids_strc,  os.path.join(cfg['common_folder'],'mycnf_fmri.cnf'),cfg)
                    QA_topup(bids_strc, 'dwi_dn_gc.nii.gz', 'dwi_dn_gc_topup.nii.gz', os.path.join(output_path, 'QA_topup'),cfg)
                    
                    extract_vols(bids_strc.get_path('dwi_dn_gc_topup.nii.gz'), bids_strc.get_path('b0_dn_gc_topup.nii.gz'), 0, 1, cfg)

                    # Generate non-deformed masks
                    if not os.path.exists(bids_strc.get_path('mask.nii.gz')) or cfg['redo_final_mask']:
                       
                        # average b0
                        make_avg(3, [bids_strc.get_path('b0_dn_gc_topup.nii.gz')], [bids_strc.get_path('b0_dn_gc_topup_avg.nii.gz')], cfg)
                        
                        # bias field correct b0
                        N4_unbias(bids_strc.get_path('b0_dn_gc_topup_avg.nii.gz'),bids_strc.get_path('b0_dn_gc_topup_avg_bc.nii.gz'), cfg)
 
                        # get brain
                        binary_op(bids_strc.get_path('b0_dn_gc_topup_avg_bc.nii.gz'),bids_strc.get_path('mask_before_preproc.nii.gz'), '-mul', bids_strc.get_path('b0_dn_gc_topup_avg_bc_brain_before_preproc.nii.gz'), cfg)

                        # register dwi --> T2w
                        antsreg_simple(bids_strc_anat.get_path('T2w_bc_brain.nii.gz'), # fixed
                                bids_strc.get_path('b0_dn_gc_topup_avg_bc_brain_before_preproc.nii.gz'),  # moving
                                bids_strc.get_path('dwiafterpreproc2T2w'), cfg)
                        
                        # apply inverse transform to put T2w in dwi space
                        ants_apply_transforms_simple([bids_strc_anat.get_path('T2w_bc_brain.nii.gz')],  # input 
                                              bids_strc.get_path('b0_dn_gc_topup_avg_bc_brain_before_preproc.nii.gz'), # moving
                                              [bids_strc.get_path('T2w_brain_in_dwiafterpreproc.nii.gz')], # output
                                              [bids_strc.get_path('dwiafterpreproc2T2w0GenericAffine.mat'), 1], cfg) # transform 1
         
                        
                        make_mask(bids_strc.get_path('T2w_brain_in_dwiafterpreproc.nii.gz'), bids_strc.get_path('mask.nii.gz'), 100, cfg)                
                        #filter_clusters_by_size(bids_strc.get_path('mask.nii.gz'), bids_strc.get_path('mask.nii.gz'), 200)
                        dilate_im(bids_strc.get_path('mask.nii.gz'), bids_strc.get_path('mask_dil.nii.gz'), '1',cfg)
        
                        # update mask brain
                        binary_op(bids_strc.get_path('b0_dn_gc_topup_avg_bc.nii.gz'),bids_strc.get_path('mask.nii.gz'), '-mul', bids_strc.get_path('b0_dn_gc_topup_avg_bc_brain.nii.gz'), cfg)

                    # Convert to mif in case
                    nifti_to_mif(bids_strc.get_path('dwi_dn_gc_topup.nii.gz'), bids_strc.get_path('bvecs_fake.txt'), bids_strc.get_path('bvalsNom.txt'), bids_strc.get_path('dwi_dn_gc_topup.mif'),cfg)

                # Quality analysis
                output_path = bids_strc.get_path();
                QA_plotSNR(bids_strc,'dwi.nii.gz', 'dwi_snr.nii.gz', 'dwi_dn_sigma.nii.gz', 'mask.nii.gz', 'bvalsNom.txt',os.path.join(output_path, 'QA_acquisition'))
                plt.close('all')
                make_summary_pdf(bids_strc.get_path(),bids_strc.get_path('summary.pdf'))

                
                    
