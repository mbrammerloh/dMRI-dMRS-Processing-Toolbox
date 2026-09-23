"""
Script to convert raw data to nifti format and organize it in BIDS format
It uses the Dicomifier module, installed in the Dicomifier python environment

Last changed Jan 2025
@author: Rita O
"""

import os
import re
import json
import sys
import pandas as pd
import math

from dmri_dmrs_toolbox.misc.custom_functions import (
    create_directory, raw_to_nifti, copy_file, extract_methods, plot_bvals
)
from dmri_dmrs_toolbox.misc.bids_structure import create_bids_structure


def Step2_raw2nii2bids(cfg):
    
    data_path       = cfg['data_path']     
    scan_list       = pd.read_excel(os.path.join(data_path, cfg['scan_list_name'] ))
    
    ######## SUBJECT-WISE OPERATIONS ########
    for subj in cfg['subj_list']:
    
        print('Converting to nifti ' + subj + '...')
    
        # Extract data for this subject
        subj_data      = scan_list[(scan_list['study_name'] == subj)].reset_index(drop=True)
        
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
            
            # Generate paths and convert data 
            raw_path        = os.path.join(data_path, 'raw_data', list(subj_data_sess['raw_data_folder'].unique())[0]) 
            nifti_path      = os.path.join(data_path, 'nifti_data', 'unsorted', subj,f"ses-{sess:02}")
            create_directory(nifti_path)
            
            scans = subj_data_sess.loc[(subj_data_sess["acqType"].isin(["PGSE", "T2W", "T1W", "STE"])),"scanNo"].tolist()
            raw_to_nifti(raw_path, nifti_path, cfg, scans)
            
            bids_strc = create_bids_structure(subj=subj, sess=sess, datatype="dwi", root=data_path, 
                                        folderlevel='nifti_data', workingdir='sorted')
            
            # Index of scans for this session 
            study_indx  = subj_data_sess.index[subj_data_sess['sessNo'] == sess].tolist()
            
            ###### SCAN-WISE OPERATIONS ######
            for scn_ctr in study_indx:
                
                # scan folder number
                scan_no = subj_data_sess['scanNo'][scn_ctr]
                
                if subj_data_sess['analyse'][scn_ctr] =='y' and subj_data_sess['acqType'][scn_ctr] == 'PGSE':
        
                    method_path   = os.path.join(raw_path,str(scan_no), 'method')
                    # Get paths and directories
                    if subj_data_sess['phaseDir'][scn_ctr] == 'fwd':
                        nii_path    = os.path.join(nifti_path,str(scan_no) + '_1_' + subj_data_sess['acqSeq'][scn_ctr])
                    elif subj_data_sess['phaseDir'][scn_ctr] == 'rev':
                        with open(os.path.join(raw_path,str(scan_no), 'acqp'), 'r') as f:
                            for line in f:
                                if '##$ACQ_scan_name=' in line: 
                                    match=re.search(r'\((.*?)\)',next(f))
                                    ref_name=match.group(1)[1:] 
                                    
                        pv = subj_data_sess['PV'][scn_ctr]
                        if  "V3.5" in pv or "V3.7" in pv:
                            nii_path    = os.path.join(nifti_path,str(scan_no) + '_1_' + 'ADJ_REVPE_E' + ref_name)
                        elif  'V1.1' in subj_data_sess['PV'][scn_ctr]:
                            nii_path    = os.path.join(nifti_path,str(scan_no) + '_1_' + subj_data_sess['acqSeq'][scn_ctr])
                    bids_strc.set_param(datatype='dwi',description='Delta_'+str(int(subj_data_sess['diffTime'][scn_ctr]))+'_'+subj_data_sess['phaseDir'][scn_ctr])

                    # Transfer files
                    create_directory(bids_strc.get_path())
                    copy_file([os.path.join(nii_path, '1.nii.gz')], [bids_strc.get_path('dwi.nii.gz')])
                    extract_methods(method_path, bids_strc, 'PGSE')
                    plot_bvals(bids_strc)
        
                elif subj_data_sess['analyse'][scn_ctr] =='y' and subj_data_sess['acqType'][scn_ctr] == 'STE':
        
                    method_path   = os.path.join(raw_path,str(scan_no), 'method')
                    # Get paths and directories
                    if subj_data_sess['phaseDir'][scn_ctr] == 'fwd':
                        nii_path    = os.path.join(nifti_path,str(scan_no) + '_1_' + subj_data_sess['acqSeq'][scn_ctr])
                    elif subj_data_sess['phaseDir'][scn_ctr] == 'rev':
                        with open(os.path.join(raw_path,str(scan_no), 'acqp'), 'r') as f:
                            for line in f:
                                if '##$ACQ_scan_name=' in line: 
                                    match=re.search(r'\((.*?)\)',next(f))
                                    ref_name=match.group(1)[1:] 
                        nii_path    = os.path.join(nifti_path,str(scan_no) + '_1_' + 'ADJ_REVPE_E' + ref_name)
                    
                    bids_strc.set_param(datatype='dwi_STE',description='STE_'+ subj_data_sess['phaseDir'][scn_ctr])
        
                    # Transfer files
                    create_directory(bids_strc.get_path())
                    copy_file([os.path.join(nii_path, '1.nii.gz')], [bids_strc.get_path('dwi.nii.gz')])
                    extract_methods(method_path, bids_strc, 'STE', cfg)
                    plot_bvals(bids_strc)
    
                elif subj_data_sess['analyse'][scn_ctr] =='y' and subj_data_sess['acqType'][scn_ctr] == 'T2W':
        
                    # Get paths and directories
                    nii_path    = os.path.join(nifti_path,str(scan_no) + '_1_' + subj_data_sess['acqSeq'][scn_ctr])
                    bids_strc.set_param(datatype='anat',description=None)

                    # Transfer files
                    create_directory(bids_strc.get_path())
                    copy_file([os.path.join(nii_path, '1.nii.gz')], [bids_strc.get_path('T2w.nii.gz')])
        
                elif subj_data_sess['analyse'][scn_ctr] =='y' and subj_data_sess['acqType'][scn_ctr] == 'B0': # probably not working
        
                    # Get paths and directories
                    nii_path    = os.path.join(nifti_path,str(scan_no)  + '_1_' + subj_data_sess['acqSeq'][scn_ctr])
                    bids_strc.set_param(datatype='B0',description=None)

                    # Transfer files
                    create_directory(bids_strc.get_path())
                    copy_file([os.path.join(nii_path, '1.nii.gz')], [bids_strc.get_path('B0.nii.gz')])
        
            #shutil.rmtree(os.path.join(data_path, 'nifti_data', 'unsorted'))


