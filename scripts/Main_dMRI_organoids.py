#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script to preprocess and analyze diffusion MRI (dMRI) data acquired on ORGANOIDS.

Current parameters are configured for organoid data by default.

The pipeline supports multi-shell acquisitions with multiple diffusion times,
including Linear Tensor Encoding (LTE) and Spherical Tensor Encoding (STE),
together with an anatomical reference image (T1- or T2-weighted).

## USAGE SUMMARY

- Prepare the cohort Excel file (see common/example_study.xlsx).

- Place raw scanner data under:
      folder_study_name/raw_data/studyName_X/

- Edit the cfg section at the top of the script before running with the 
  desired processing parameters.

- Run individual steps (StepX) to inspect outputs (recommended),
  or run the full pipeline (not recommmended).
  

## ORGANOID-SPECIFIC NOTES 

Organoid processing follows the same general pipeline as other datasets, with two
important exceptions during Step3_preproc:

1) Manual organoid masks are required.
   The pipeline will prompt you to create masks defining individual organoids.
   This is currently done manually (e.g., using FSLeyes) by drawing one mask per
   organoid and saving the files in the anatomical folder with names such as:
       organoidA_mask.nii.gz, organoidB_mask.nii.gz, ...

   These masks are then used to extract parameter estimates for each these 
   organoid regions.

2) Manual registration using 3D Slicer is required.
   Registration for organoid data is performed via landmark-based registration
   in 3D Slicer. During preprocessing, the pipeline will prompt you to open
   3D Slicer and perform this step manually.
   

Last changed Feb 2026
@author: Rita O
"""

import os
import matplotlib.pyplot as plt
from dmri_dmrs_toolbox.misc.custom_functions import update_cfg
from dmri_dmrs_toolbox.dwi.Step1_fill_study_excel import Step1_fill_study_excel
from dmri_dmrs_toolbox.dwi.Step2_raw2nii2bids import Step2_raw2nii2bids
from dmri_dmrs_toolbox.dwi.Step3_preproc_anat import Step3_preproc_anat
from dmri_dmrs_toolbox.dwi.Step3_preproc_dwi import Step3_preproc_dwi
from dmri_dmrs_toolbox.dwi.Step3_registrations import Step3_registrations
from dmri_dmrs_toolbox.dwi.Step4_modelling import Step4_modelling
from dmri_dmrs_toolbox.dwi.Step5_get_estimates import Step5_get_estimates

plt.close('all');
os.system('clear')
os.system('cls')

########################## SCRIPT CONFIGURATION (EDIT AS APPPROPRIATE) ##########################

#### DATA PATH AND SUBJECTS ####
subj_list = ['org-B4-D32-6','org-B4-D32-9',
             'org-B4-D34-7','org-B4-D34-12','org-B4-D34-14','org-B4-D34-16','org-B4-D34-18','org-B4-D34-21',
             'org-B5-D6-3',
             'org-B5-D16-3','org-B5-D16-9',
             'org-B5-D30-4',
             'org-B5-D38-3','org-B5-D38-8','org-B5-D38-12','org-B5-D38-16','org-B5-D38-22','org-B5-D38-26','org-B5-D38-31','org-B5-D38-35','org-B5-D38-39','org-B5-D38-44',
             'org-B5-D40-2',
             'org-B5-D44-4','org-B5-D44-9'] 

cfg                         = {}
cfg['subj_list']            = subj_list
cfg["sess_list"]            = [1]
cfg['data_path']            = os.path.join(os.path.expanduser('~'), 'Documents','Rita','Data','data_organoids_April2026')          # path to where the data from the cohort is
cfg['toolboxes']            = os.path.join(os.path.expanduser('~'),  'Documents','Rita','Toolboxes')                              # path to where some toolboxes from matlab are (including MPPCA and tMPPCA)
cfg['prep_foldername']      = 'preprocessed'    # name of the preprocessed folder (keep 'preprocessed' as default)
cfg['analysis_foldername']  = 'analysis'        # name of the analysis folder (keep 'analysis' as default)
cfg['common_folder']        = os.path.join(os.path.expanduser('~'), 'Documents','Rita','Data','common')  # path to the common folder with files needed throught the pipeline
cfg['scan_list_name']       = 'ScanList.xlsx'   # name of the excel file containing the metadata of the cohort (fill in before some of the columns)
cfg['atlas']                = 'anat_space_organoids'      # name of the brain atlas to be used in the analysis. This atlas needs to exists in the common folder
cfg['atlas_TPM']            = []      # name of the tissue probability map (tpm) to be used to threshold GM and WM to define more precisly the ROIs. This atlas needs to exists in the common folder

#### DWI PREPROCESSING CONFIG #### 
cfg['do_topup']             = 1   # If there is data to do topup correction, set to 1, otherwise to 0
cfg['redo_all']             = 0   # If you want to redo everything set to 1 (will delete previous results)
cfg['redo_bet_anat']        = 0   # If you want to redo the processing from the brain extraction set to 1
cfg['redo_b0_extract']      = 0   # If you want to redo the processing from the dwi b0 extraction set to 1
cfg['redo_merge_dwi']       = 0   # If you want to redo the processing from the merging of dwi files from different diffusion times set to 1
cfg['redo_firstmask']       = 0   # If you want to redo the processing from the initial brain mask on the combined dataset, set this to 1
cfg['redo_denoise']         = 0   # If you want to redo the processing from the denoising set to 1
cfg['redo_gibbs']           = 0   # If you want to redo the processing from the gibbs unringing set to 1
cfg['redo_topup']           = 0   # If you want to redo the processing from the topup correction set to 1
cfg['redo_eddy']            = 0   # If you want to redo the processing from the eddy correction set to 1
cfg['redo_final_mask']      = 0   # If you want to redo the processing from the creation of the final brain masks set to 1

cfg['algo_denoising']       = 'matlab_MPPCA'         # Options are: 'matlab_MPPCA', or 'matlab_tMPPCA_4D' or 'matlab_tMPPCA_5D' or 'mrtrix_MPPCA' or 'designer_tMPPCA'. Note that designer sigma output map is not caculated the same as for the other methods
#cfg['algo_brainextract']    = 'RATS'                # Options are: 'BET' or 'RATS' Not needed for organoids
cfg['anat_thr']             = '1200'                 # intensity threshold for the anatomical image
cfg['anat_format']          = 'T2w'                  # Depends on you anatomical image. Common options are: 'T1w' or 'T2w'
cfg['subject_type']         = 'organoid'             # Options are: 'human' or 'rat' or 'organoid'
cfg['is_alive']             = 'ex_vivo'              # Options are: 'in_vivo' or 'ex_vivo'
cfg['individual_rev']       = 1                      # If there is one rev direction acquired for each diffusion time write 1, otherwise 0
cfg['acq_wholesphere']      = 0                      # If data is acquired on the whole sphere (1) or not (0). Important for eddy correction.
cfg['topup_cfg_name']       = 'mycnf_fmri.cnf'  # name of the file with parameter details for topup (should be in the common folder)

#### DWI MODEL CONFIG ####
cfg['model_list_GM']        =  []         # List of model names to use for fitting the GM signal
cfg['model_list_WM']        =  []               # List of model names to use for fitting the WM signal
cfg['LTEDelta_for_microFA'] =  38               # Diffusion time (in ms) from the LTE (linear tensor encoding) acquisition that needs to be used together with STE (spherical tensor encoding) data to compute microFA
cfg['redo_modelling']       =  0                # If you want to redo the modelling set to 1, otherwise will redo just the models that didnt do before (because it crashed or something)

#### ROIS CONFIG ####
cfg['ROIs_GM']       = ['organoids']    # List of ROIs to analyse (in GM). Defined previously for each atlas in custom_functions. Please read instructions of Step3_registrations
cfg['ROIs_WM']       = []    # List of ROIs to analyse (in WM). Defined previously for each atlas in custom_functions. Please read instructions of Step3_registrations
cfg['ROIs_manual']   = []                   # List of ROIs to analyse (in WM). Defined previously for each atlas in atlas_functions. Keep empty [] if desired.

cfg['tpm_thr']       = 0.8   # Threshold to be used for the tissue probability map (tpm) to define the different tissues
cfg['mrs_vx']        = 0                        # Does the dataset include mrs. 1 if yes, 0 if no
cfg['lat_ROIS']      = 0                        # Do you want to have ROIs in left and right hemispheres separately? 1 if yes, 0 if no. It requires adding a column VoxMidHem in the excel with the voxel of the middle plane that separates the hemisphere for each subject. It assumes a given orientation in the data order so it might not work for human and organoid data.

#### SOFTWARES ####
cfg["conda_exe"]     = "conda"   # Environment manager tool. Options are "conda" or "micromamba" 
cfg["ants_path"]     = "/home/localadmin/SOFTWARES/ants-2.5.3/bin"    # path to ANTS
cfg["fsl_path"]      = "/home/localadmin/fsl/bin"                     # path to FSL
cfg["mrtrix_path"]   = "/home/localadmin/anaconda3/bin"               # path to mrtrix
cfg["rats_path"]     = "/home/localadmin/SOFTWARES/Rodent_Seg/distribution2/" # path to RATS_MM
cfg['use_server_mount'] = 0  # Set to 1 if data is on a server-mounted filesystem that Docker cannot mount.
                             # Data will be copied locally before running Docker.
                             # Note: if the code itself is also running from the server mount, this option will not help.

cfg['slice_dim'] = 2  # dimention of the image to plot


#### SAVE CONFIG FILE ####
cfg = update_cfg(cfg)


########################## DATA PROCESSING ##########################

#### STEP 1. COHORT DEFINITION ####
Step1_fill_study_excel(cfg)   

#### STEP 2. NIFTI CONVERT SUBJECT  ####

# 2.1 Use Dicomifier to convert to nifi (pass as argument the datapath to load the cfg file)
Step2_raw2nii2bids(cfg) 

# 2.2 Correct orientation from bruker system to be consisten with normal atlas and everything else
#Step2_correct_orientation(cfg)  

#### STEP 3. PREPROCESS SUBJECT ####

# 3.1 Process normal linear encoding data with PSGE (LTE), along with anatomical image
Step3_preproc_anat(cfg) 
Step3_preproc_dwi(cfg) 

# 3.2 Process spherical encoding (STE) data, assumes an anatomical image has already been processed before
# Only needed if STE data is present
#from Step3_preproc_STE import *
#Step3_preproc_STE(cfg) 

# 3.3 Register atlas to anatomical image and then to dwi for individual or combined diffusion times. 
# If exists also fits STE to LTE. Does not register across sessions for now
# If exists also fits STE to LTE. Does not register across sessions for now
# Takes a long time. comment out if no regional estimations are needed
Step3_registrations(cfg)

#### STEP 4. MODELLING SUBJECT ####

# Fit the dwi signal with models like Nexi, Sandi, SMI, ....
# Diffusion Tensor (DTI) and Kurtusis (DKI) is always done by default. 
# To be faster and perform only DTI and DKI fitting just leave cfg['model_list_GM'] 
# and cfg['model_list_WM'] empty.
Step4_modelling(cfg)

#### STEP 5. GET VALUES ####

# Retreives parameter estimates from the model fits, making summary figures and excel with data in certain ROIs
# Needs the registration step 3 to be done before.
Step5_get_estimates(cfg) 
