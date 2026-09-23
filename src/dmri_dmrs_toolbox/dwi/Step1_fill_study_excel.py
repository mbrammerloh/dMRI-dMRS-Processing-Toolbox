"""
Script to fill in an excel with information from th study and imaging data
It does not use a particular python environment.

Last changed Jan 2025
@author: Rita O
"""

import pandas as pd
import os
import re

def Step1_fill_study_excel(cfg):
    
    data_path       = cfg['data_path']      
    file_path       = os.path.join(data_path, cfg['scan_list_name'] )
    list_methods    = pd.read_excel(file_path)
    
    # Add new columns
    required_cols = ['PV', 'phaseDir', 'acqSeq', 'diffTime', 'noDirs', 'noBval', 'noB0', 'NR', 'NA',
                     'noDummy','SE','EPIfact','diffDuration', 'noShapes', 'DurGrad1', 'DurGrad2']
    
    for col in required_cols:
        if col not in list_methods.columns:
            list_methods[col] = None  # create new column
    list_methods[col] = list_methods[col].astype('object')
    list_methods['PV'] = list_methods['PV'].astype('object')
    list_methods['phaseDir'] = list_methods['phaseDir'].astype('object')
    list_methods['acqSeq'] = list_methods['acqSeq'].astype('object')

    for ii in range (list_methods.shape[0]):
        
        if list_methods.at[ii, 'acqType'] not in ['PGSE', 'STE', 'T2W', 'T1W']:
            continue  # if not one of the known acquisitions, skip
    
        scan_path   = os.path.join(data_path, 'raw_data', list_methods['raw_data_folder'][ii], str(list_methods['scanNo'][ii]))
    
        # Extract the name of the sequence  
        with open(os.path.join(scan_path, 'acqp'), 'r') as f:
            for line in f:
                if '##$ACQ_protocol_name=' in line:
                    #seq_name = line.split(':')[1][:-2]
                    match=re.search(r'\<(.*?)\>',next(f))
                    seq_name=match.group(1) 
                    if seq_name == 'RARE':
                        list_methods.at[ii, 'acqSeq'] = 'T2_Turbo' + seq_name
                    elif seq_name == 'FieldMap':
                        list_methods.at[ii, 'acqSeq'] = 'B0Map-ADJ_B0MAP'
                    else: # for ihMT and the diffusion sequence the sequence name is ok                   
                        list_methods.at[ii, 'acqSeq'] = seq_name
        
        # Extract methods for the dMRI sequence
        if seq_name.find('DTI_EPI') != -1 or seq_name.find('DtiEpi') != -1  or seq_name.find('dtiEpi') != -1 or seq_name.find('REVPE') != -1:
    
            with open(os.path.join(scan_path, 'method'), 'r') as f: 
                for line in f:
    
                    if '##TITLE=' in line:
                        pv_version = line.split(',')[1]
                        list_methods.at[ii, 'PV'] = pv_version.strip()
    
                    if '##$PVM_DwGradSep=( 1 )' in line:
                        diff_time = next(f)
                        list_methods.at[ii, 'diffTime'] = float(diff_time)
    
                    if '##$ReversePEAdjRunning='  in line or 'EPI_YN_ForwardReverseMode' in line or 'PVM_EpiReversedPE' in line:
                        rev_opt = line.split('=')[1]
                        if rev_opt.strip() == 'No':
                            list_methods.at[ii, 'phaseDir'] = 'fwd'
                        elif rev_opt.strip() == 'Yes':
                            list_methods.at[ii, 'phaseDir'] = 'rev'
    
                    if '##$PVM_DwNDiffDir=' in line:
                        no_dirs = line.split('=')[1]
                        list_methods.at[ii, 'noDirs'] = int(no_dirs)
    
                    if '##$PVM_DwNDiffExpEach=' in line:
                        no_bval = line.split('=')[1]
                        list_methods.at[ii, 'noBval'] = int(no_bval)
    
                    if '##$PVM_DwAoImages=' in line:
                        no_b0 = line.split('=')[1]
                        list_methods.at[ii, 'noB0'] = int(no_b0)
    
                    if '##$PVM_NRepetitions=' in line:
                        no_rep = line.split('=')[1]
                        list_methods.at[ii, 'NR'] = int(no_rep)
    
                    if '##$PVM_NAverages=' in line:
                        no_avg = line.split('=')[1]
                        list_methods.at[ii, 'NA'] = int(no_avg)
    
                    if '##$PVM_DummyScans=' in line:
                        no_dummy = line.split('=')[1]
                        list_methods.at[ii, 'noDummy'] = int(no_dummy)
    
                    if '##$PVM_EpiEchoSpacing=' in line:
                        SE = line.split('=')[1]
                        list_methods.at[ii, 'SE'] = float(SE)
    
                    if '##$PVM_EpiNEchoes=' in line:
                        n_echoes = line.split('=')[1]
                        list_methods.at[ii, 'EPIfact'] = int(n_echoes)
                    
                    if '##$PVM_DwGradDur=( 1 )' in line:
                        diff_dur = next(f)
                        list_methods.at[ii, 'diffDuration'] = float(diff_dur)
       
        elif seq_name.find('my_DOR_R12_EPI') != -1:
            with open(os.path.join(scan_path, 'method'), 'r') as f: 
                for line in f:
                    if '##TITLE=' in line:
                        pv_version = line.split(',')[1]
                        list_methods.at[ii, 'PV'] = pv_version.strip()
                    if '$$ /opt/PV' in line:
                        folder = line.split('/method')[0]
                        folder = folder.split('/')[-1]
                        if list_methods['scanNo'][ii]==int(folder):
                            list_methods.at[ii, 'phaseDir'] = 'fwd'
                        else:
                            list_methods.at[ii, 'phaseDir'] = 'rev'
                    if '##$DwShapeDirVec=' in line:
                        no_dirs = line.split('=')[1]
                        list_methods.at[ii, 'noDirs'] = int(no_dirs[2])
                    if '##$DwNShapes=' in line:
                        no_shapes = line.split('=')[1]
                        list_methods.at[ii, 'noShapes'] = int(no_shapes)
                    if '##$DwGradAmpG=' in line:
                        no_amp = line.split('=')[1]
                        list_methods.at[ii, 'noBval'] = int(no_amp[2])
                    if '##$ProtRep=' in line:
                        no_rep = line.split('=')[1]
                        list_methods.at[ii, 'NR'] = int(no_rep)
                    if '##$PVM_NAverages=' in line:
                        no_avg = line.split('=')[1]
                        list_methods.at[ii, 'NA'] = int(no_avg)
                    if '##$PVM_DummyScans=' in line:
                        no_dummy = line.split('=')[1]
                        list_methods.at[ii, 'noDummy'] = int(no_dummy)
                    if '##$DwMinModuleDur=' in line:
                        grad1_dur = line.split('=')[1]
                        list_methods.at[ii, 'DurGrad1'] = float(grad1_dur)
                        list_methods.at[ii, 'DurGrad2'] = float(grad1_dur)
                        
        else: # for all the other sequences, extract only the PV version
    
            with open(os.path.join(scan_path, 'method'), 'r') as f: 
                for line in f:
    
                    if '##TITLE=' in line:
                        pv_version = line.split(',')[1]
                        list_methods.at[ii, 'PV'] = str(pv_version.strip())
    
    df              = pd.DataFrame(list_methods)
    df.to_excel(file_path, index=False) 