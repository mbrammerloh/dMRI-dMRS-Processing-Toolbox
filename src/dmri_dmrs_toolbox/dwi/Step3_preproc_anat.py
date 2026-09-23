"""
Script to preprocess dMRI data.

Includes: 
- Denoising. several options exist
    Options are: 'matlab_MPPCA' - uses matlab and performs normal MPPCA on matrix with this format [x, y, z, bvals/bvecs x diffTime] (4D)
                 'matlab_tMPPCA_4D' - uses matlab and performs tMPPCA on matrix with this format [x, y, z, bvals/bvecs x diffTime] (4D)
                 'matlab_tMPPCA_5D'- uses matlab and performs tMPPCA on matrix with this format [x, y, z, bvals/bvecs, diffTime] (5D)
                 'mrtrix_MPPCA' - uses mrtrix to perform MPPCA on matrix with this format [x, y, z, bvals/bvecs x diffTime] (4D)
                 'designer_tMPPCA' - uses mrtrix to perform tMPPCA on matrix with this format [x, y, z, bvals/bvecs x diffTime] (4D)
    Note that designer sigma output map is not caculated on the software, so it's calculated here by hand but it's the same formula as for the other methods.
    Note that the implementation of MPPCA on mrtrix and on matlab are slightly different (matlab version denoises more the data and creates smoother maps)
    Note that for the 5D version you have to have a complete matrix (this is with the same number of diffusion times for each bvals/bvecs)
- Gibbs correction with mrtrix 
- Top up with FSL
- Eddy with FSL

It runs for the combined dataset (all diffusion times) or 
for each diffusion time alone.

It does not use a particular python environment (base).
It plots some quality analysis (QA) results for checking.

Last changed June 2025
@author: Rita O
"""

import os
import sys
import pandas as pd
import math
import subprocess
from dmri_dmrs_toolbox.misc.custom_functions import (fsl_reorient, N4_unbias, brain_extract_BET, brain_extract_RATS, interactive_brain_mask_refine, QA_brain_extract, make_mask, copy_files, fsl_mult)
from dmri_dmrs_toolbox.misc.bids_structure import create_bids_structure
from dmri_dmrs_toolbox.misc.atlas_functions import make_atlas_manual_organoid, make_atlas_label_from_anat_organoid
import matplotlib.pyplot as plt
import shutil
plt.close('all')

def Step3_preproc_anat(cfg):
    
    data_path   = cfg['data_path']     
    scan_list   = pd.read_excel(os.path.join(data_path, cfg['scan_list_name']))
    anat_format = cfg['anat_format']

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
    
        ######## SESSION-WISE OPERATIONS ########
        for sess in sess_list:
            
            subj_data_sess = subj_data[subj_data["sessNo"] == sess]
            print('Working on session ' + str(sess) + '...')
            
            # Copy nifti data to preprocessing folder
            nifti_path      = os.path.join(data_path, 'nifti_data', 'sorted', subj,f"ses-{sess:02}",'anat')
            preproc_path    = os.path.join(data_path, 'derivatives', cfg['prep_foldername'], subj,f"ses-{sess:02}",'anat')
            if not os.path.exists(preproc_path) or cfg['redo_all']:
                if os.path.exists(preproc_path):
                    print("Your previous results will be deleted, are you sure? Press any key to continue or abort.")
                    input()
                    shutil.rmtree(preproc_path)
                
                shutil.copytree(nifti_path, preproc_path)

            ########################## ANATATOMICAL PROCESSING ##########################
            bids_strc_anat = create_bids_structure(subj=subj, sess=sess, datatype="anat", root=data_path, 
                                        folderlevel='derivatives', workingdir=cfg['prep_foldername'])
            
            # BET
            if not os.path.exists(bids_strc_anat.get_path(f'{anat_format}_bc_brain.nii.gz')) or cfg['redo_bet_anat']:
                print('Processing anatomical data')
                
                # Check that data exists
                if not os.path.exists(bids_strc_anat.get_path(f'{anat_format}.nii.gz')):
                    print("No anat scans found — exiting.")
                    return
                
                # If human data ensure it's oriented in a standard way
                if cfg['subject_type']=='human':
                    fsl_reorient(bids_strc_anat.get_path(f'{anat_format}.nii.gz'))
                    
                # Correct for bias field   
                N4_unbias(bids_strc_anat.get_path(f'{anat_format}.nii.gz'),bids_strc_anat.get_path(f'{anat_format}_bc.nii.gz'),cfg)
                
                # Brain extraction
                if cfg['subject_type']=='human':
                    brain_extract_BET(bids_strc_anat.get_path(f'{anat_format}_bc.nii.gz'),cfg)
                    
                elif cfg['subject_type']=='rat':
                    
                    # Using RATS
                    if cfg['algo_brainextract']=='RATS':
                        # create intial brian mask
                        brain_extract_RATS(bids_strc_anat.get_path(f'{anat_format}_bc.nii.gz'),cfg)
                        
                        # refine mask (interactive)
                        new_thr = interactive_brain_mask_refine(bids_strc_anat.get_path(f'{anat_format}_bc.nii.gz'), subj_data_sess, cfg)

                    elif cfg['algo_brainextract']=='UNET':
                        # create intial brain mask
                        print('\n >> Running U-net to do brain extraction. It takes a while...')
                        cmd = [cfg["conda_exe"], "run", "-n", "RodentSkullStrip",
                            "rbm",
                            bids_strc_anat.get_path(f'{anat_format}_bc.nii.gz'),
                            bids_strc_anat.get_path(f'{anat_format}_bc_brain_mask.nii.gz')]
                        subprocess.run(cmd, check=True)
                      
                        # refine mask (interactive)
                        new_thr = interactive_brain_mask_refine(bids_strc_anat.get_path(f'{anat_format}_bc.nii.gz'), subj_data_sess, cfg)
                        
                    
                    elif cfg['algo_brainextract']=='thr':
                        # create intial brain mask
                        print('\n >> Thresholding brain map for brain extraction.')
                        
                        # get initial threshold for the anatomical image
                        mask = subj_data_sess['acqType'].str.contains('T2W|T1W', case=False)
                        anat_thr = float(subj_data_sess.loc[mask, 'anat_thr'].iloc[0])
                      
                        make_mask(bids_strc_anat.get_path(f'{anat_format}_bc.nii.gz'), 
                                  bids_strc_anat.get_path(f'{anat_format}_bc_brain_mask.nii.gz'), anat_thr, cfg)

                        # refine mask (interactive)
                        new_thr = interactive_brain_mask_refine(bids_strc_anat.get_path(f'{anat_format}_bc.nii.gz'), subj_data_sess, cfg)

                    print('Saving new anat threshold to the excel')
                    mask_scan =  (scan_list['study_name'] == subj) & (scan_list['acqType'].str.contains('T2W|T1W', case=False))
                    scan_list.loc[mask_scan, 'anat_thr'] = new_thr
                    scan_list.to_excel(os.path.join(data_path, cfg['scan_list_name']), index=False)
                 
                    # add lesions masks if needed
                    # if subj_data_sess['group'].unique()=='KI':
                    #     make_mask(bids_strc_anat.get_path(f'{anat_format}_bc_brain.nii.gz'), bids_strc_anat.get_path('lesion_mask.nii.gz'), 20000, cfg)
                    #     filter_clusters_by_size(bids_strc_anat.get_path('lesion_mask.nii.gz'),bids_strc_anat.get_path('lesion_mask.nii.gz'),500)
                    #     erode_im_fsl(bids_strc_anat.get_path(f'{anat_format}_bc_brain_mask.nii.gz'),bids_strc_anat.get_path(f'{anat_format}_bc_brain_mask_ero.nii.gz'),cfg)
                    #     fsl_mult(bids_strc_anat.get_path('lesion_mask.nii.gz'),bids_strc_anat.get_path(f'{anat_format}_bc_brain_mask_ero.nii.gz'),bids_strc_anat.get_path('lesion_mask.nii.gz'),cfg)
                    #     create_inverse_mask(bids_strc_anat.get_path('lesion_mask.nii.gz'),bids_strc_anat.get_path(f'{anat_format}_bc_brain_mask.nii.gz'),bids_strc_anat.get_path(),cfg)
                
                elif cfg['subject_type']=='organoid':
                    
                    ## Make mask of the overall organoids -------------
                    print('\n >> Thresholding brain map for brain extraction.')
                    
                    # get initial threshold for the anatomical image
                    mask = subj_data_sess['acqType'].str.contains('T2W|T1W', case=False)
                    anat_thr = float(subj_data_sess.loc[mask, 'anat_thr'].iloc[0])
                  
                    make_mask(bids_strc_anat.get_path(f'{anat_format}_bc.nii.gz'), 
                              bids_strc_anat.get_path(f'{anat_format}_bc_brain_mask.nii.gz'), anat_thr, cfg)

                    # refine mask (interactive)
                    new_thr = interactive_brain_mask_refine(bids_strc_anat.get_path(f'{anat_format}_bc.nii.gz'), subj_data_sess, cfg)

                    print('Saving new anat threshold to the excel')
                    mask_scan =  (scan_list['study_name'] == subj) & (scan_list['acqType'].str.contains('T2W|T1W', case=False))
                    scan_list.loc[mask_scan, 'anat_thr'] = new_thr
                    scan_list.to_excel(os.path.join(data_path, cfg['scan_list_name']), index=False)
                    
                    copy_files([bids_strc_anat.get_path(f'{anat_format}_bc_brain_mask.nii.gz')],[bids_strc_anat.get_path('organoids_mask.nii.gz')])    
                    
                    ## Make mask of the the flask that would be like "brain" mask -------------
                    make_mask(bids_strc_anat.get_path(f'{anat_format}_bc.nii.gz'), 
                              bids_strc_anat.get_path(f'{anat_format}_bc_brain_mask.nii.gz'), 5000, cfg)
                    fsl_mult(bids_strc_anat.get_path(f'{anat_format}_bc.nii.gz'),bids_strc_anat.get_path(f'{anat_format}_bc_brain_mask.nii.gz'),bids_strc_anat.get_path(f'{anat_format}_bc_brain.nii.gz'),cfg)
                    
                    # Make label of the atlas
                    make_atlas_label_from_anat_organoid(bids_strc_anat.get_path('organoids_mask.nii.gz'),
                                               bids_strc_anat.get_path(),
                                               bids_strc_anat.get_path(f"{cfg['atlas']}.label"), cfg)   
                    
                    # Make manually some masks of the organoids
                    prompt = (      
                       "\n==================== Organoid Mask Definition ====================\n"
                        "A preliminary mask covering all organoids in the flask has already been "
                        "generated automatically using an intensity threshold.\n\n"
                        
                        "If you require a more accurate segmentation, you can MANUALLY DRAW MASKS "
                        "for your selected organoids.\n\n"
                         
                        "If you would like to create or refine the organoid masks now, this is the time to do so.\n"
                        "Otherwise, type 'no' to skip this step.\n\n"
                        
                        "To create the masks, follow these steps:\n\n"
    
                        "1) Open FSLeyes.\n"
                        "   Load the anatomical image:\n"
                        "     {anat_format}_bc.nii.gz\n\n"
                        "2) Create organoid masks:\n"
                        "   - Go to: Tools → Edit mode → Create empty 3D mask\n"
                        "   - Use the pencil tool to manually paint ONE organoid.\n"
                        "   - Save the mask in NIfTI format in the anatomical folder.\n"
                        "   - Naming convention:\n"
                        "       sub-X_ses-X_organoidA_mask.nii.gz\n"
                        "       sub-X_ses-X_organoidB_mask.nii.gz\n"
                        "       sub-X_ses-X_organoidC_mask.nii.gz\n"
                        "     (use A, B, C, ... for multiple organoids)\n\n"
                        "3) Repeat mask creation for all organoids you want to analyse.\n\n"
                        "4) When finished:\n"
                        "   - Return to this terminal\n"
                        "   - Type 'yes' to continue\n\n"
                        "NOTE:\n"
                        "During parameter extraction, the pipeline will automatically\n"
                        "search for files named '*_organoidX_mask.nii.gz' and extract\n"
                        "estimates within each region.\n"
                        ">> Did you create your mask? Type 'yes' or 'no':\n"
                        "=================================================================\n"
                    )

                    
                    # apply inverse transform to put T2w in dwi space
                    answer = input(prompt).strip().lower()
                    if answer == 'yes':
                        make_atlas_manual_organoid(bids_strc_anat.get_path('organoids_manual_mask.nii.gz'),
                                                   bids_strc_anat.get_path(),
                                                   bids_strc_anat.get_path(f"{cfg['atlas']}.label"), cfg)    
                        print('done')  
                    elif answer == 'no':  
                        print("Continuing without masks.")                                      
                    else:
                        print("Aborted by user.")
                        sys.exit("Pipeline stopped by user input.")
                        
    
                # QA
                QA_brain_extract(bids_strc_anat.get_path(f'{anat_format}_bc.nii.gz'),os.path.join(bids_strc_anat.get_path(),'QA_brain_extract'),anat_format, cfg)
