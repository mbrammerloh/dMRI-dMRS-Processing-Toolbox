#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script to analyse dMRS data

Last changed Jan 2025
@author: Rita, Malte
"""

import os
import matplotlib.pyplot as plt
import json
from dmri_dmrs_toolbox.dmrs.Step1_preproc import Step1_preproc
from dmri_dmrs_toolbox.dmrs.Step2_fitting import Step2_fitting
from pathlib import Path
import numpy as np

plt.close('all');
os.system('clear')
os.system('cls')

########################## SCRIPT CONFIGURATION (EDIT AS APPPROPRIATE) ##########################

#### DATA PATH AND SUBJECTS ####
subj_list = [f'sub-{i:02d}' for i in list(np.arange(3,29))]    # list of subjects to analyse [8,10,11,12,14,15]]#
subj_list = [f'sub-{i:02d}' for i in list(np.arange(20,29))]    # list of subjects to analyse [8,10,11,12,14,15]]#

cfg                         = {}
cfg['subj_list']            = subj_list
cfg['data_path']            = "/media/localadmin/DATA/data/CTD/"          # path to where the data from the cohort is
cfg['prep_foldername']      = 'preprocessed'    # name of the preprocessed folder (keep 'preprocessed' as default)
cfg['analysis_foldername']  = 'analysis'        # name of the analysis folder (keep 'analysis' as default)
cfg['common_folder']        = "/home/localadmin/Software/dMRI-dMRS-Processing-Toolbox/common/"  # path to the common folder with files needed throught the pipeline
cfg['scan_list_name']       = 'ScanList_CTD.xlsx'   # name of the excel file containing the metadata of the cohort

#### DMRS PREPROCESSING CONFIG ####
cfg['basis_set'] = os.path.join(cfg['common_folder'], 'mrs_basis_sets',
                                'Basis_Set_dSPECIAL_differentTM')  # path to where the basis set are
cfg['models'] = ["dti", "stick", "dki", "cylinder", "cylinder_sphere", "sphere_stick"]  # models used for fitting
cfg['metabolites'] = ['NAA+NAAG', 'Glu', 'Ins', 'GPC+PCho', 'Cr+PCr', 'Tau', 'Gln']  # metabolites for analysis
cfg['redo_processing'] = 1  # 1 to remove previous file and redo all processing (Step1); 0 to process only missing TMs

#### SOFTWARES ####
cfg['toolboxes']            = "/home/localadmin/Software/"                              # path to where some toolboxes from matlab are (including MPPCA and tMPPCA)
cfg['LC_model_path']             = os.path.join(cfg['toolboxes'], 'LCModel','binaries','linux')  # path to LC model executable
cfg['MATLAB_Runtime']       = "/home/localadmin/Software/Matlab_runtime/R2025a" # path to MATLAB_Runtime. It needs to be R2025a: https://ch.mathworks.com/products/compiler/matlab-runtime.html

#### SAVE CONFIG FILE ####
with open(cfg['data_path'] + '/.config_mrs.json', 'w') as f:
    json.dump(cfg, f)

#### STEP 1. Process and quantify bruker data
Step1_preproc(cfg)

#### STEP 2. Fitting of data
Step2_fitting(cfg)