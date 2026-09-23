"""
Script to retrieve model estimates within regions of interest.
Needs registration of an atlas to work.
Compatible with any Python environment.

IMPORTANT!!: 
 If new atlas arrives please go to atlas_functions.py and edit:
    prepare_atlas, create_ROI_mask and prepare_atlas_labels functions
 to account for the ROIs and atlas that you want.

Last updated: Jan 2025
@author: Rita O
"""

import os
import glob
import math
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
from dmri_dmrs_toolbox.misc.bids_structure import create_bids_structure
from dmri_dmrs_toolbox.misc.custom_functions import *
from dmri_dmrs_toolbox.misc.atlas_functions import prepare_atlas_labels, create_ROI_mask
from dmri_dmrs_toolbox.misc.custom_functions_models import get_param_names_model

plt.close('all')


def Step5_get_estimates(cfg):
    data_path = cfg['data_path']
    scan_list = pd.read_excel(os.path.join(data_path, cfg['scan_list_name'] ))
    cfg['model_list'] = cfg['model_list_GM'] + cfg['model_list_WM'] + cfg['ROIs_manual']
    import distinctipy
    color_list = distinctipy.get_colors(len(cfg['ROIs_GM'] + cfg['ROIs_WM'] + cfg['ROIs_manual'])+2, pastel_factor=0.5)
    color_list_snr  =  [(0.3462032775519162, 0.3531070236388303, 0.8723545410491512), (0.4324776646215159, 0.9594894437563749, 0.33498906309524773), (0.9406821354408894, 0.3893640567923122, 0.3692878637990247), (0.4072131417883373, 0.8783467338575792, 0.9732166499618664), (0.883784919700095, 0.5084984818886036, 0.9831871536574889), (0.9725395306324506, 0.954894866579424, 0.37771765906993093)]

    ######## SUBJECT-WISE OPERATIONS ########
    for subj in cfg['subj_list']:
        print(f'Getting model estimates for {subj}...')
        
        subj_data = scan_list[scan_list['study_name'] == subj].reset_index(drop=True)
        subj_data = subj_data[subj_data['analyse'] == 'y']
        
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
            print(f'Session {sess}...')
            
            filtered_data = subj_data_sess[
                (subj_data_sess['phaseDir'] == 'fwd') &
                (subj_data_sess['sessNo'] == sess) &
                (subj_data_sess['noBval'] > 1) &
                (subj_data_sess['acqType'] == 'PGSE') 
            ]
            Delta_list = sorted(filtered_data['diffTime'].dropna().astype(int).unique())
            if cfg['lat_ROIS']==1:
                vx_middle = filtered_data['VoxMidHem'].dropna().astype(int).unique()[0]
            else:
                vx_middle = 0
                print('You are not going to have lateralized ROIs. Please ignore the files _left and _right because they are not valid in this case')
            
            ######## MODEL-WISE OPERATIONS ########
            for model in cfg['model_list']:
                print(f'Getting model estimates from {model}...')

                filtered_data = subj_data_sess[(subj_data_sess['acqType'] == 'PGSE') &
                                          (subj_data_sess['phaseDir'] == 'fwd') &
                                          (subj_data_sess['sessNo'] == sess) &
                                          (subj_data_sess['noBval'] > 1)]
                
               
                bids_strc_analysis = create_bids_structure(subj=subj, sess=sess, datatype='dwi', root=data_path, 
                             folderlevel='derivatives', workingdir=cfg['analysis_foldername'], description=model)
                output_path = os.path.join(bids_strc_analysis.get_path(), 'Output_masked')

                bids_strc_reg  = create_bids_structure(subj=subj, sess=sess, datatype='registration', description=cfg['atlas']+'-To-'+'allDelta-allb', root=data_path, 
                                             folderlevel='derivatives', workingdir=cfg['analysis_foldername'])
                bids_strc_reg.set_param(base_name='')
                
                # Define atlas
                atlas = bids_strc_reg.get_path('atlas_in_dwi.nii.gz')

                ######## OPERATIONS INVOLVING THE NEED OF AN ATLAS  ########
                # Define atlas labels 
                if cfg['atlas']:
                    if 'anat_space_organoids' not in  cfg['atlas'] :
                        atlas_labels = prepare_atlas_labels(cfg['atlas'], glob.glob(os.path.join(cfg['common_folder'], cfg['atlas'], '*label*'))[0])
                    else:
                        bids_strc_anat = create_bids_structure(subj=subj, sess=sess, datatype='anat', root=data_path, 
                                                   folderlevel='derivatives', workingdir=cfg['prep_foldername'])   
                        atlas_labels = prepare_atlas_labels(cfg['atlas'], glob.glob(os.path.join(bids_strc_anat.get_path(), '*label*'))[0])
                else:
                    atlas_labels=''
                    
                # Define TPMs
                if cfg['atlas_TPM']:
                    bids_strc_reg_TPM  = create_bids_structure(subj=subj, sess=sess, datatype='registration', description=cfg['atlas_TPM']+'-To-'+'allDelta-allb', root=cfg['data_path'] , 
                                                 folderlevel='derivatives', workingdir=cfg['analysis_foldername'])
                    bids_strc_reg_TPM.set_param(base_name='')
                    TPMs = []
                    for tissue in ['GM', 'WM', 'CSF']:
                        path = bids_strc_reg_TPM.get_path(f'atlas_TPM_{tissue}_in_dwi.nii.gz')
                        TPMs.append(path if os.path.exists(path) else '')
                else:
                    TPMs = []

                # Determine ROI list and output file
                if model in cfg['model_list_GM']:
                    ROI_list = cfg['ROIs_GM'].copy() 
                    outfile = os.path.join(os.path.dirname(os.path.dirname(output_path)), f"output_ROIs_{cfg['atlas']}_GM_{model}.xlsx")
                    outfile2 = os.path.join(os.path.dirname(os.path.dirname(output_path)), f"output_ROIs_{cfg['atlas']}_GM_{model}.npy")

                else:
                    ROI_list = cfg['ROIs_WM'].copy() 
                    outfile = os.path.join(os.path.dirname(os.path.dirname(output_path)), f"output_ROIs_{cfg['atlas']}_WM_{model}.xlsx")
                    outfile2 = os.path.join(os.path.dirname(os.path.dirname(output_path)), f"output_ROIs_{cfg['atlas']}_WM_{model}.npy")

                # If there is mrs voxel, add it as ROI
                bids_mrs  = create_bids_structure(subj=subj, sess=sess, datatype='registration', description=f"dmrs-to-allDelta-allb", root=data_path, 
                                               folderlevel='derivatives', workingdir=cfg['analysis_foldername'])
                bids_mrs.set_param(base_name='')
                if cfg['mrs_vx'] == 1 and os.path.exists(bids_mrs.get_path()):
                    ROI_list.append('voxel_mrs');
                    if TPMs:
                        ROI_list.append('voxel_mrs_GM')
                        
                # If there is a lesion mask, add it as a ROI
                if cfg['ROIs_manual']:
                    ROI_list.extend(cfg['ROIs_manual'])
                    if TPMs:
                        ROI_list.extend(cfg['ROIs_manual'])
                        
                # If there is manual masks
                bids_manual = create_bids_structure(subj=subj, sess=sess, datatype='', root=data_path,
                        folderlevel='derivatives', workingdir='manual_masks')
                bids_manual.set_param(description='')
                        
                ######## EXTRACT MODEL ESTIMATES ########
                # Option 1. Extract estimates with user defined ROIs
                if os.path.exists(output_path):
                    
                    # Initialize variables
                    patterns, lims, maximums = get_param_names_model(model,cfg['is_alive'])
                    if model != 'DTI_DKI' and  model != 'Micro_FA':
                        model_clean  = model.split('_')[0]
                    cleaned_patterns = [re.sub(r'\*{2,}', '*', re.sub(model_clean, '', p, flags=re.IGNORECASE).replace('[^s]', '')).strip('*') for p in patterns]          
                   
                    # Get values of parameters inside the ROIs        
                    Data, Data_all, Data_l, Data_r = get_values_within_ROI(
                        ROI_list, atlas, atlas_labels, TPMs, cfg['tpm_thr'], 
                        vx_middle, patterns, maximums, bids_strc_reg, bids_mrs, bids_manual, output_path)
                    
                    # Create table structure with results
                    df_data = pd.DataFrame(Data, columns=cleaned_patterns)
                    df_data.insert(0, 'ROI Name', ROI_list)
                
                    # Option 2. Extract estimates from all labels in the atlas (dont save nifiti files of mask)
                    # Personally think it's not very relevant
                    # Data2 = np.zeros((len(atlas_labels['IDX'].to_numpy()), len(patterns)))
                    # for i, indx in enumerate(atlas_labels['IDX'].to_numpy()):
                    #     mask = create_ROI_mask_fromindx(atlas, atlas_labels, indx, bids_strc_reg)

                    #     for j, (pattern, maximum) in enumerate(zip(patterns, maximums)):
                    #         param_img = nib.load(glob.glob(os.path.join(output_path, pattern))[0]).get_fdata()
                    #         masked = param_img[mask > 0]  # Select only voxels inside the ROI
                    #         #masked_clean = masked[~np.isnan(masked) & (masked > maximum[0]) & (masked < maximum[1])]
                    #         masked_clean = masked[~np.isnan(masked)]
                    #         Data2[i, j] = np.median(masked_clean) if len(masked_clean) > 0 else np.nan

                    # Add mean of medians
                    #df_data.loc[len(df_data)] = ['mean of medians'] + np.nanmean(Data2, axis=0).tolist()
                    
                    # Save summary of means in excel format
                    df_data.to_excel(outfile, index=False)

                    # Save all data in npy format
                    df_data_all = pd.DataFrame(Data_all, columns=cleaned_patterns)
                    df_data_all.insert(0, 'ROI Name', ROI_list)
                    np.save(outfile2, df_data_all)
                    
                    if cfg['lat_ROIS']==1:
                        df_data_l = pd.DataFrame(Data_l, columns=cleaned_patterns)
                        df_data_l.insert(0, 'ROI Name', ROI_list)
                        np.save(outfile2.replace('.npy','_left.npy'), df_data_l)
                                
                        df_data_r = pd.DataFrame(Data_r, columns=cleaned_patterns)
                        df_data_r.insert(0, 'ROI Name', ROI_list)
                        np.save(outfile2.replace('.npy','_right.npy'), df_data_r)
                    
                    ######## Plot estimates in bar graph ########
                    num_patterns = len(patterns)
                    num_ROIs = len(ROI_list)
                    fig, axes = plt.subplots(num_patterns, 1, figsize=(10, 4 * num_patterns), sharex=True)
                    
                    for param_idx in range(num_patterns):  
                        ax = axes[param_idx]  
                       
                        # plot data
                        x = np.arange(num_ROIs)  
                        y = Data[:, param_idx]  
                        ax.bar(x, y, width=0.4, color='grey')
                    
                        # polish
                        paramname = cleaned_patterns[param_idx]
                        ax.set_ylabel(f'{paramname}', fontsize=12, fontweight='bold')
                        if param_idx == 1:
                            ax.set_ylim(1, 4)
                        elif param_idx == 2:
                            ax.set_ylim(0.5, 1.5)
                        elif param_idx == 3:
                            ax.set_ylim(0, 1)
                        ax.grid(True)
                        ax.set_axisbelow(True)
                        
                    # yaxis, titles and save figure           
                    axes[-1].set_xticks(x)
                    axes[-1].set_xticklabels(ROI_list, rotation=45, ha='right')
                    plt.suptitle(model, fontsize=14, fontweight='bold')
                    plt.subplots_adjust(wspace=0.05,hspace=0.05, top=0.95, bottom=0.1, left=0.1, right=0.90) 
                    outfile = os.path.join(os.path.dirname(os.path.dirname(output_path)), f"output_ROIs_{cfg['atlas']}_GM_{model}.png")
                    plt.savefig(outfile, bbox_inches='tight', dpi=300)

                    ######## PLOT SNR ########

                    if model == 'Nexi':
                        print(f'Plotting powder average signal within ROIs...')

                        # Load data
                        bvals = read_numeric_txt(os.path.join(bids_strc_analysis.get_path(),'powderaverage.bval'))
                        S_S0  = nib.load(os.path.join(bids_strc_analysis.get_path(),'powderaverage_dwi.nii.gz')).get_fdata()
                        nf = nib.load(os.path.join(bids_strc_analysis.get_path(),'normalized_sigma.nii.gz')).get_fdata()*np.sqrt(np.pi/2)
                        WB_mask = nib.load(os.path.join(bids_strc_analysis.get_path(),'updated_mask.nii.gz')).get_fdata()
                        
                        # organize
                        bvals_split = np.split(bvals[0], len(Delta_list))

                        # Loop through ROIs  
                        temp_ROI = copy.deepcopy(ROI_list)
                        temp_ROI.append('WB_mask')
                        
                        n_params = len(temp_ROI)
                        n_rows = 1 if n_params <= 4 else 2
                        n_cols = math.ceil(n_params / n_rows)
                        
                        fig, axs = plt.subplots(n_rows, n_cols, figsize=(8, 4))
                        if len(temp_ROI) != 1:
                            axs = axs.flatten()
                        fig.subplots_adjust(wspace=0.05, hspace=0.18, top=0.92, bottom=0.15, left=0.1, right=0.95)

                        if len(temp_ROI) == 1:
                                axs = [axs]  # ensure axs is always iterable
                        k=0
                          
                        for ROI in temp_ROI:
     
                            if ROI == 'voxel_mrs':
                                mask_indexes = nib.load(bids_strc_reg.get_path('mask_voxel_mrs.nii.gz')).get_fdata()
                            elif ROI == 'voxel_mrs_GM':
                                mask_indexes = nib.load(bids_strc_reg.get_path('mask_voxel_mrs_GM.nii.gz')).get_fdata()
                            elif os.path.exists(bids_manual.get_path(f'mask_{ROI}.nii.gz')):
                                 mask_indexes = nib.load(bids_manual.get_path(f'mask_{ROI}.nii.gz')).get_fdata() > 0
                            elif ROI =='WB_mask':
                                mask_indexes = WB_mask>0
                            else:
                                mask_indexes = create_ROI_mask(atlas, atlas_labels, TPMs, ROI, cfg['tpm_thr'], bids_strc_reg)
                            
                            S_S0_masked = copy.deepcopy(S_S0)
                            for v in range(S_S0_masked.shape[-1]):
                                S_S0_masked[:, :, :, v] = np.multiply(S_S0_masked[:, :, :, v], mask_indexes)
                                
                            data = S_S0_masked.reshape(S_S0_masked.shape[0]*S_S0_masked.shape[1]*S_S0_masked.shape[2], S_S0_masked.shape[3])
                            data = data[~(np.isnan(data).any(axis=1) | (data == 0).any(axis=1))]
                            data_split = np.split(data, len(Delta_list), axis=1)
                            
                            nf_masked = nf * mask_indexes
                            nf_masked = nf_masked.reshape(nf_masked.shape[0]*nf_masked.shape[1]*nf_masked.shape[2], 1)
                            nf_masked = nf_masked[~(np.isnan(nf_masked).any(axis=1) | (nf_masked == 0).any(axis=1))]
    
                            # Plot data
                            for i in range(len(Delta_list)):
                                axs[k].plot(
                                    np.transpose(bvals_split[i]),
                                    np.nanmean(data_split[i], axis=0),
                                    marker='o',
                                    linestyle='--',
                                    color=color_list_snr[i],
                                    label=f'$\\Delta$= {Delta_list[i]}',
                                    markersize=3
                                )
                            axs[k].plot(np.transpose(bvals_split[0]), np.repeat(np.nanmean(nf_masked), np.transpose(bvals_split[0]).shape[0]),color='black',label='nf')

                            # Settings
                            #axs[k].set_yscale('log')
                            row = k // n_cols
                            col = k % n_cols
                            axs[k].set_ylim([0.02, 0.7])
                            axs[k].set_xticks(bvals_split[i])
                            axs[k].tick_params(axis='both', which='major', labelsize=8,length=4)
                            #axs[k].set_yticks([0.02, 0.1, 1])
                            if col == 0:
                               axs[k].set_ylabel(r'$S / S_0$', fontdict={'size': 9,  'style': 'italic'})
                               if row==0:
                                   axs[k].legend(fontsize=8)
                            else:
                                axs[k].set_yticklabels([])
                            if row == n_rows -1:
                               axs[k].set_xlabel(r'$b$ $[ms/µm^2]$', fontdict={'size': 9,  'style': 'italic'})
                               #axs[k].set_xticklabels(np.round(bvals_split[i]).astype(int))
                            else:
                               axs[k].set_xticklabels([])
                            axs[k].grid(True)
                            axs[k].set_title(ROI, fontsize=10)
                            k += 1
                        n_used = n_params 
                        if len(axs) > n_used:
                             for ax in axs[n_used:]:
                                 ax.set_visible(False)  
                       
                        plt.savefig(os.path.join(os.path.dirname(os.path.dirname(output_path)), 'SignalDecay_summary.png'))
                        plt.close(fig)
                        
                    ######## PLOT Signal vs b^-1/2 ########

                    if model == 'Nexi':
                        print(f'Plotting powder average signal within ROIs against 1/sqrt(b)...')

                        # Load data
                        bvals = read_numeric_txt(os.path.join(bids_strc_analysis.get_path(),'powderaverage.bval'))
                        S_S0  = nib.load(os.path.join(bids_strc_analysis.get_path(),'powderaverage_dwi.nii.gz')).get_fdata()
                        WB_mask = nib.load(os.path.join(bids_strc_analysis.get_path(),'updated_mask.nii.gz')).get_fdata()

                        # organize
                        bvals_split = np.split(bvals[0], len(Delta_list))

                        # Loop through ROIs   
                        temp_ROI = copy.deepcopy(ROI_list)
                        temp_ROI.append('WB_mask')
                        n_params = len(temp_ROI)
                        n_rows = 1 if n_params <= 4 else 2
                        n_cols = math.ceil(n_params / n_rows)
                        
                        fig, axs = plt.subplots(n_rows, n_cols, figsize=(8, 4))
                        if len(temp_ROI) != 1:
                           axs = axs.flatten()
                        fig.subplots_adjust(wspace=0.05, hspace=0.48, top=0.90, bottom=0.19, left=0.1, right=0.95)

                        if len(temp_ROI) == 1:
                                axs = [axs]  # ensure axs is always iterable
                        k=0
                            
                        for ROI in temp_ROI:
     
                            # Plot data
                            if ROI == 'voxel_mrs':
                                mask_indexes = nib.load(bids_strc_reg.get_path('mask_voxel_mrs.nii.gz')).get_fdata()
                            elif ROI == 'voxel_mrs_GM':
                                mask_indexes = nib.load(bids_strc_reg.get_path('mask_voxel_mrs_GM.nii.gz')).get_fdata()
                            elif os.path.exists(bids_manual.get_path(f'mask_{ROI}.nii.gz')):
                                 mask_indexes = nib.load(bids_manual.get_path(f'mask_{ROI}.nii.gz')).get_fdata() > 0
                            elif ROI =='WB_mask':
                                mask_indexes = WB_mask>0
                            else:
                                mask_indexes = create_ROI_mask(atlas, atlas_labels, TPMs, ROI, cfg['tpm_thr'], bids_strc_reg)
    
                            S_S0_masked = copy.deepcopy(S_S0)
                            for v in range(S_S0_masked.shape[-1]):
                                S_S0_masked[:, :, :, v] = np.multiply(S_S0_masked[:, :, :, v], mask_indexes)
        
                            data = S_S0_masked.reshape(S_S0_masked.shape[0]*S_S0_masked.shape[1]*S_S0_masked.shape[2], S_S0_masked.shape[3])
                            data = data[~(np.isnan(data).any(axis=1) | (data == 0).any(axis=1))]
                            data_split = np.split(data, len(Delta_list), axis=1)
                            
                            # Plot data
                            for i in range(len(Delta_list)):
                                axs[k].plot(
                                    1 / np.sqrt(bvals_split[i]),
                                    np.nanmean(data_split[i], axis=0),
                                    marker='o',
                                    linestyle='--',
                                    color=color_list_snr[i],
                                    label=f'$\\Delta$= {Delta_list[i]}',
                                    markersize=3
                                )
                                
                            #axs[k].set_yscale('log')
                            row = k // n_cols
                            col = k % n_cols
                            axs[k].set_ylim([0.02, 0.7])
                            axs[k].set_xticks(1 / np.sqrt(bvals_split[i]))
                            axs[k].tick_params(axis='both', which='major', labelsize=8,length=4)
                            #axs[k].set_yticks([0.02, 0.1, 1])
                            if col == 0:
                               axs[k].set_ylabel(r'$S / S_0$', fontdict={'size': 9, 'weight': 'bold', 'style': 'italic'})
                               if row==0:
                                   axs[k].legend(fontsize=8)
                            else:
                                axs[k].set_yticklabels([])
                            if row == n_rows -1:
                               axs[k].set_xlabel(r'$1/b^{-1/2}$ $[µm/ms^{1/2}]$', fontdict={'size': 9, 'weight': 'bold', 'style': 'italic'})
                               axs[k].set_xticks( 1 / np.sqrt(bvals_split[i]))
                               axs[k].set_xticklabels(np.round(1 / np.sqrt(bvals_split[i]), 2))
                               axs[k].set_xticklabels(axs[k].get_xticklabels(), rotation=45)
                               ax_top = axs[k].twiny()
                               ax_top.set_xlim(axs[k].get_xlim())
                               tick_positions = 1 / np.sqrt(bvals_split[i])
                               ax_top.set_xticks(tick_positions)
                               ax_top.set_xticklabels(np.round(bvals_split[i], 1))
                               ax_top.tick_params(axis='both', which='major', labelsize=8,length=4)


                            else:
                               axs[k].set_xticklabels([])
                            axs[k].grid(True)
                            axs[k].set_title(ROI, fontsize=10)
                            k += 1
                        n_used = n_params 
                        if len(axs) > n_used:
                             for ax in axs[n_used:]:
                                 ax.set_visible(False)
                                 
                        plt.savefig(os.path.join(os.path.dirname(os.path.dirname(output_path)), 'SignalDecay_summary2.png'))
                        plt.close(fig)
                       
            ######## PLOT Signal for pwd ########

            print(f'Plotting powder average signal within ROIs...')
            
            bids_strc_reg  = create_bids_structure(subj=subj, sess=sess, datatype='registration', description=cfg['atlas']+'-To-'+'allDelta-allb', root=data_path, 
                                         folderlevel='derivatives', workingdir=cfg['analysis_foldername'])
            bids_strc_reg.set_param(base_name='')
            
            # Define atlas
            atlas = bids_strc_reg.get_path('atlas_in_dwi.nii.gz')
            
            ######## OPERATIONS INVOLVING THE NEED OF AN ATLAS  ########
            # Define atlas labels
            if cfg['atlas']:
                if 'anat_space_organoids' not in  cfg['atlas'] :
                    atlas_labels = prepare_atlas_labels(cfg['atlas'], glob.glob(os.path.join(cfg['common_folder'], cfg['atlas'], '*label*'))[0])
                else:
                    bids_strc_anat = create_bids_structure(subj=subj, sess=sess, datatype='anat', root=data_path, 
                                               folderlevel='derivatives', workingdir=cfg['prep_foldername'])   
                    atlas_labels = prepare_atlas_labels(cfg['atlas'], glob.glob(os.path.join(bids_strc_anat.get_path(), '*label*'))[0])
            else:
                atlas_labels=''
                
            # Define TPMs
            if cfg['atlas_TPM']:
                bids_strc_reg_TPM  = create_bids_structure(subj=subj, sess=sess, datatype='registration', description=cfg['atlas_TPM']+'-To-'+'allDelta-allb', root=cfg['data_path'] , 
                                             folderlevel='derivatives', workingdir=cfg['analysis_foldername'])
                bids_strc_reg_TPM.set_param(base_name='')
                TPMs = []
                for tissue in ['GM', 'WM', 'CSF']:
                    path = bids_strc_reg_TPM.get_path(f'atlas_TPM_{tissue}_in_dwi.nii.gz')
                    TPMs.append(path if os.path.exists(path) else '')
            else:
                TPMs = []
                
            delta_used = Delta_list[0]
            bids_strc_analysis = create_bids_structure(subj=subj, sess=sess, datatype='dwi', root=data_path,
                folderlevel='derivatives', workingdir=cfg['analysis_foldername'], description=f'pwd_avg_Delta_{delta_used}')
            output_path = os.path.join(bids_strc_analysis.get_path())
    
            # Load data
            bvals = read_numeric_txt(get_file_in_folder(bids_strc_analysis,'*bvalsNom_avg.txt'))
            S_S0  = nib.load(get_file_in_folder(bids_strc_analysis,'*pwd_avg.nii.gz')).get_fdata()

            # organize
            ROI_list = cfg['ROIs_GM'].copy() + cfg['ROIs_WM'].copy() + cfg['ROIs_manual'].copy()  #+ ['WB_mask']

            # Loop through ROIs     
            n_params = len(ROI_list)
            n_rows = 1 if n_params <= 4 else 2
            n_cols = math.ceil(n_params / n_rows)
            
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(8, 4))
            if len(ROI_list) != 1:
                axs = axs.flatten()
            fig.subplots_adjust(wspace=0.05, hspace=0.18, top=0.92, bottom=0.15, left=0.1, right=0.95)

            if len(ROI_list) == 1:
                    axs = [axs]  # ensure axs is always iterable
            k=0
                
            for ROI in ROI_list:
 
                if ROI == 'voxel_mrs':
                    mask_indexes = nib.load(bids_mrs.get_path('voxel_mrs.nii.gz')).get_fdata()
                elif ROI == 'voxel_mrs_GM':
                    mask_indexes = nib.load(bids_mrs.get_path('voxel_mrs_GM.nii.gz')).get_fdata()
                elif os.path.exists(bids_manual.get_path(f'mask_{ROI}.nii.gz')):
                     mask_indexes = nib.load(bids_manual.get_path(f'mask_{ROI}.nii.gz')).get_fdata() > 0
                elif ROI =='WB_mask':
                    mask_indexes = WB_mask>0
                else:
                    mask_indexes = create_ROI_mask(atlas, atlas_labels, TPMs, ROI, cfg['tpm_thr'], bids_strc_reg)
                
                S_S0_masked = copy.deepcopy(S_S0)
                for v in range(S_S0_masked.shape[-1]):
                    S_S0_masked[:, :, :, v] = np.multiply(S_S0_masked[:, :, :, v], mask_indexes)
                    
                data = S_S0_masked.reshape(S_S0_masked.shape[0]*S_S0_masked.shape[1]*S_S0_masked.shape[2], S_S0_masked.shape[3])
                data = data[~(np.isnan(data).any(axis=1) | (data == 0).any(axis=1))]
                data_split = np.split(data, len(bvals), axis=1)
                

                # Plot data
                axs[k].plot(
                    np.transpose(bvals),
                    np.nanmean(data, axis=0),
                    marker='o',
                    linestyle='--',
                    color=color_list_snr[0],
                    label=f'$\\Delta$= {delta_used}',
                    markersize=3
                )

                # Settings
                #axs[k].set_yscale('log')
                row = k // n_cols
                col = k % n_cols
                #axs[k].set_ylim([0.02, 0.7])
                axs[k].set_xticks(bvals[0])
                #axs[k].set_yticks([0.02, 0.1, 1])
                if col == 0:
                   axs[k].set_ylabel(r'$S / S_0$', fontdict={'size': 10, 'weight': 'bold', 'style': 'italic'})
                   if row==0:
                       axs[k].legend()
                else:
                    axs[k].set_yticklabels([])
                if row == n_rows -1:
                   axs[k].set_xlabel(r'$b$ $[ms/µm^2]$', fontdict={'size': 10, 'weight': 'bold', 'style': 'italic'})
                  # axs[k].set_xticklabels(np.round(bvals[0]).astype(int))
                else:
                   axs[k].set_xticklabels([])
                axs[k].grid(True)
                axs[k].set_title(ROI)
                k += 1
            n_used = n_params 
            if len(axs) > n_used:
                 for ax in axs[n_used:]:
                     ax.set_visible(False)  
           
            plt.savefig(os.path.join((os.path.dirname(output_path)), 'SignalDecay_pwd_summary.png'))
            plt.close(fig)
            
            ######## EXTRACT DTI,DKI ESTIMATES ########
            ######## OPERATIONS INVOLVING THE NEED OF AN ATLAS  ########

            ROI_list = cfg['ROIs_GM'] + cfg['ROIs_WM'] + cfg['ROIs_manual']
            
            patterns, lims, maximums = get_param_names_model('DTI_DKI',cfg['is_alive'])
            cleaned_patterns = [re.sub(r'\*{2,}', '*', re.sub('DTI_DKI', '', p, flags=re.IGNORECASE).replace('[^s]', '')).strip('*') for p in patterns]          
            
            # If there is mrs voxel, add it as ROI
            bids_mrs  = create_bids_structure(subj=subj, sess=sess, datatype='registration', description=f"dmrs-to-allDelta-allb", root=data_path, 
                                           folderlevel='derivatives', workingdir=cfg['analysis_foldername'])
            bids_mrs.set_param(base_name='')
            if cfg['mrs_vx'] == 1 and os.path.exists(bids_mrs.get_path()):
                 ROI_list.append('voxel_mrs'); 
                 if TPMs:
                     ROI_list.append('voxel_mrs_GM')
                     
            # If there is manual masks
            bids_manual = create_bids_structure(subj=subj, sess=sess, datatype='', root=data_path,
                  folderlevel='derivatives', workingdir='manual_masks')
            bids_manual.set_param(description='')
            
            Data_DTIDKI    = np.empty((len(Delta_list), len(ROI_list), len(patterns)), dtype=object)
            Data_DTIDKI_l  = np.empty((len(Delta_list), len(ROI_list), len(patterns)), dtype=object)
            Data_DTIDKI_r  = np.empty((len(Delta_list), len(ROI_list), len(patterns)), dtype=object)

            # Loop through deltas
            for d_idx, Delta in enumerate(Delta_list):
                print(f'Getting model estimates from DTI/DKI for Delta={Delta}...')

                data_used = f'Delta_{Delta}'
                bids_strc_analysis = create_bids_structure(subj=subj, sess=sess, datatype='dwi', root=data_path,
                    folderlevel='derivatives', workingdir=cfg['analysis_foldername'], description=f'DTI_DKI_{data_used}')
                output_path = os.path.join(bids_strc_analysis.get_path(), 'Output_masked')
            
                bids_strc_reg  = create_bids_structure(subj=subj, sess=sess, datatype='registration', description=cfg['atlas']+'-To-'+'allDelta-allb', root=data_path, 
                                         folderlevel='derivatives', workingdir=cfg['analysis_foldername'])
                bids_strc_reg.set_param(base_name='')
            
                # Define atlas  
                atlas = bids_strc_reg.get_path('atlas_in_dwi.nii.gz')
            
                # Define atlas labels 
                if cfg['atlas']:
                    if 'anat_space_organoids' not in  cfg['atlas']:
                        atlas_labels = prepare_atlas_labels(cfg['atlas'], glob.glob(os.path.join(cfg['common_folder'], cfg['atlas'], '*label*'))[0])
                    else:
                        bids_strc_anat = create_bids_structure(subj=subj, sess=sess, datatype='anat', root=data_path, 
                                                   folderlevel='derivatives', workingdir=cfg['prep_foldername'])   
                        atlas_labels = prepare_atlas_labels(cfg['atlas'], glob.glob(os.path.join(bids_strc_anat.get_path(), '*label*'))[0])
                else:
                    atlas_labels=''
                    
                # Define TPMs
                if cfg['atlas_TPM'] :
                    bids_strc_reg_TPM  = create_bids_structure(subj=subj, sess=sess, datatype='registration', description=cfg['atlas_TPM']+'-To-'+'allDelta-allb', root=cfg['data_path'] , 
                                             folderlevel='derivatives', workingdir=cfg['analysis_foldername'])
                    bids_strc_reg_TPM.set_param(base_name='')
                    TPMs = []
                    for tissue in ['GM', 'WM', 'CSF']:
                        path = bids_strc_reg_TPM.get_path(f'atlas_TPM_{tissue}_in_dwi.nii.gz')
                        TPMs.append(path if os.path.exists(path) else '')
                else:
                    TPMs = []
                             
                # Get data
                if cfg['atlas'] != '' and not os.path.exists(atlas):
                    raise FileNotFoundError(f"Atlas file not found: {atlas}")

                if os.path.exists(output_path):
                        
                    # Get values of parameters inside the ROIs        
                    Data, Data_all, Data_all_l, Data_all_r = get_values_within_ROI(
                        ROI_list, atlas, atlas_labels, TPMs, cfg['tpm_thr'], 
                        vx_middle, patterns, maximums, bids_strc_reg, bids_mrs, bids_manual, output_path)
                    
                    Data_DTIDKI[d_idx, :, :]     = Data_all
                    Data_DTIDKI_l[d_idx, :, :]   = Data_all_l
                    Data_DTIDKI_r[d_idx, :, :]   = Data_all_r
                                    
                # Save summary of means in excel format
                df_data = []
                df_data = pd.DataFrame( np.vectorize(np.nanmean, otypes=[float])(Data_all), columns=cleaned_patterns)
                df_data.insert(0, 'ROI Name', ROI_list)
                outfile = os.path.join(os.path.dirname(os.path.dirname(output_path)), f"output_ROIs_{cfg['atlas']}_DTI_DKI_Delta_{Delta}.xlsx")
                df_data.to_excel(outfile, index=False)
                
                # Save all data in npy format
                outfile2 = os.path.join(os.path.dirname(os.path.dirname(output_path)), f"output_ROIs_{cfg['atlas']}_DTI_DKI_Delta_{Delta}.npy")
                df_data_all = pd.DataFrame(Data_all, columns=cleaned_patterns)
                df_data_all.insert(0, 'ROI Name', ROI_list)
                np.save(outfile2, df_data_all)
                
                if cfg['lat_ROIS']==1:
                    df_data_l = pd.DataFrame(Data_all_l, columns=cleaned_patterns)
                    df_data_l.insert(0, 'ROI Name', ROI_list)
                    np.save(outfile2.replace('.npy','_left.npy'), df_data_l)
                            
                    df_data_r = pd.DataFrame(Data_all_r, columns=cleaned_patterns)
                    df_data_r.insert(0, 'ROI Name', ROI_list)
                    np.save(outfile2.replace('.npy','_right.npy'), df_data_r)
                
            # Plot results
            if os.path.exists(atlas) and os.path.exists(output_path):
                
                # Start figure
                fig, ax = plt.subplots(3, 3, figsize=(9, 6))
                fig.subplots_adjust(wspace=0.02, hspace=0.10, top=0.91, bottom=0.14, left=0.2, right=0.9)
                
                # Main figure
                line_handles = []
                for i, ROI in enumerate(ROI_list):
                    y0 =  np.vectorize(np.nanmean, otypes=[float])(Data_DTIDKI[:, i, 0])
                    s0 =  np.vectorize(np.nanstd, otypes=[float])(Data_DTIDKI[:, i, 0])
                    y1 =  np.vectorize(np.nanmean, otypes=[float])(Data_DTIDKI[:, i, 1])
                    s1 =  np.vectorize(np.nanstd, otypes=[float])(Data_DTIDKI[:, i, 1])
                    y2 =  np.vectorize(np.nanmean, otypes=[float])(Data_DTIDKI[:, i, 2])
                    s2 =  np.vectorize(np.nanstd, otypes=[float])(Data_DTIDKI[:, i, 2])

                    line = plot_with_shade(ax[0,0], Delta_list, y0, s0, color_list[i],
                           linestyle='--', marker='o')
                    plot_with_shade(ax[1,0], Delta_list, y1, s1, color_list[i],
                                    linestyle='--', marker='o')
                    plot_with_shade(ax[2,0], Delta_list, y2, s2, color_list[i],
                                    linestyle='--', marker='o')
                    ax[0, 0].set_title('All', fontsize=12, fontweight='bold')
                    
                    
                    y0 =  np.vectorize(np.nanmean, otypes=[float])(Data_DTIDKI_l[:, i, 0])
                    s0 =  np.vectorize(np.nanstd, otypes=[float])(Data_DTIDKI_l[:, i, 0])
                    y1 =  np.vectorize(np.nanmean, otypes=[float])(Data_DTIDKI_l[:, i, 1])
                    s1 =  np.vectorize(np.nanstd, otypes=[float])(Data_DTIDKI_l[:, i, 1])
                    y2 =  np.vectorize(np.nanmean, otypes=[float])(Data_DTIDKI_l[:, i, 2])
                    s2 =  np.vectorize(np.nanstd, otypes=[float])(Data_DTIDKI_l[:, i, 2])
                    
                    plot_with_shade(ax[0,1], Delta_list, y0, s0, color_list[i],
                    linestyle='--', marker='o')
                    plot_with_shade(ax[1,1], Delta_list, y1, s1, color_list[i],
                                    linestyle='--', marker='o')
                    plot_with_shade(ax[2,1], Delta_list, y2, s2, color_list[i],
                                    linestyle='--', marker='o')
                    ax[0, 1].set_title('Left', fontsize=12, fontweight='bold')
                    
                    y0 =  np.vectorize(np.nanmean, otypes=[float])(Data_DTIDKI_r[:, i, 0])
                    s0 =  np.vectorize(np.nanstd, otypes=[float])(Data_DTIDKI_r[:, i, 0])
                    y1 =  np.vectorize(np.nanmean, otypes=[float])(Data_DTIDKI_r[:, i, 1])
                    s1 =  np.vectorize(np.nanstd, otypes=[float])(Data_DTIDKI_r[:, i, 1])
                    y2 =  np.vectorize(np.nanmean, otypes=[float])(Data_DTIDKI_r[:, i, 2])
                    s2 =  np.vectorize(np.nanstd, otypes=[float])(Data_DTIDKI_r[:, i, 2])
                    
                    plot_with_shade(ax[0,2], Delta_list, y0, s0, color_list[i],
                    linestyle='--', marker='o')
                    plot_with_shade(ax[1,2], Delta_list, y1, s1, color_list[i],
                                    linestyle='--', marker='o')
                    plot_with_shade(ax[2,2], Delta_list, y2, s2, color_list[i],
                                    linestyle='--', marker='o')
                    ax[0, 2].set_title('Right', fontsize=12, fontweight='bold')
    
                   
                    line_handles.append(line)
                
                # Put everything with the same ylim
                for (i, j), a in np.ndenumerate(ax):
                    
                    if cfg['lat_ROIS']==1:
                        m1=np.vectorize(np.nanmean, otypes=[float])(Data_DTIDKI[:, :, i])
                        s1=np.vectorize(np.nanstd, otypes=[float])(Data_DTIDKI[:, :, i])
                        m2=np.vectorize(np.nanmean, otypes=[float])(Data_DTIDKI_l[:, :, i])
                        s2=np.vectorize(np.nanstd, otypes=[float])(Data_DTIDKI_l[:, :, i])
                        m3=np.vectorize(np.nanmean, otypes=[float])(Data_DTIDKI_r[:, :, i])
                        s3=np.vectorize(np.nanstd, otypes=[float])(Data_DTIDKI_r[:, :, i])

                        temp = np.min([np.nanmin(m1-s1), np.nanmin(m2-s2), np.nanmin(m3-s3)])
                        lower = round(np.floor(temp/0.05) * 0.05,2)
                        temp = np.min([np.nanmax(m1+s1), np.nanmax(m2+s2), np.nanmax(m3+s3)])
                        upper = round(np.ceil(temp/0.05) * 0.05,2)
                    else:
                        m=np.vectorize(np.nanmean, otypes=[float])(Data_DTIDKI[:, :, i])
                        s=np.vectorize(np.nanstd, otypes=[float])(Data_DTIDKI[:, :, i])
                        lower = round(np.floor(np.nanmin(m-s)/0.05) * 0.05,2)
                        upper = round(np.ceil(np.nanmax(m+s)/0.05) * 0.05,2)
                    if np.isnan(lower):
                        lower = 0
                    if np.isnan(upper):
                        upper = 1
                                            
                    a.set_ylim([lower, upper])
                    a.set_yticks([lower, upper])
                    
                # Set axis and labels for first column
                ax[0,0].set_ylabel('$MD$'); ax[0,0].tick_params(labelbottom=False)
                ax[1,0].set_ylabel('$MK$'); ax[1,0].tick_params(labelbottom=False)
                ax[2,0].set_ylabel('$FA$'); ax[2,0].set_xlabel('Diffusion time [ms]')
                
                # Remove y-axis labels and ticks for 2nd and 3rd columns
                for col in (1, 2):
                    for r in range(3):
                        ax[r, col].set_yticklabels([])   
                        ax[r, col].set_ylabel('')        
                        ax[r, col].tick_params(labelleft=False, labelbottom=False)  
                ax[2,1].set_xlabel('Diffusion time [ms]'); ax[2,2].set_xlabel('Diffusion time [ms]')
                ax[2,1].tick_params(labelbottom=True); ax[2,2].tick_params(labelbottom=True)  
               
                # Remove axis completly if there is no left/right information
                if cfg['lat_ROIS'] == 0:
                    for col in (1, 2):  
                        for r in (0,1,2):
                            ax[r, col].remove()
            
                # Plot legends, title and save
                plt.legend(handles=line_handles, labels=ROI_list, loc='center left', fontsize=7, bbox_to_anchor=(0.99, 0.5))
                plt.suptitle('DKI')
                plt.rc('font', size=9)
                plt.savefig(os.path.join(os.path.dirname(os.path.dirname(output_path)), 'DKI.png'))
                plt.close(fig)
              
                
            ######## EXTRACT MicroFA ESTIMATES ########
            ######## OPERATIONS INVOLVING THE NEED OF AN ATLAS  ########
            ROI_list = cfg['ROIs_GM'] + cfg['ROIs_WM'] + cfg['ROIs_manual']

            bids_STE      = create_bids_structure(subj=subj, sess=sess, datatype='dwi_STE', root=cfg['data_path'] , 
                          folderlevel='derivatives', workingdir=cfg['analysis_foldername'],description='microFA')        
            output_path = os.path.join(bids_STE.get_path())
            bids_strc_reg  = create_bids_structure(subj=subj, sess=sess, datatype='registration', description=cfg['atlas']+'-To-'+'allDelta-allb', root=data_path, 
                                          folderlevel='derivatives', workingdir=cfg['analysis_foldername'])
            bids_strc_reg.set_param(base_name='')

            # Define atlas
            atlas = bids_strc_reg.get_path('atlas_in_dwi.nii.gz')

            if os.path.exists(atlas) and os.path.exists(bids_STE.get_path()):
                print(f'Getting model estimates from Micro FA...')

                # Define atlas labels 
                if cfg['atlas']:
                    if 'anat_space_organoids' not in  cfg['atlas'] :
                        atlas_labels = prepare_atlas_labels(cfg['atlas'], glob.glob(os.path.join(cfg['common_folder'], cfg['atlas'], '*label*'))[0])
                    else:
                        bids_strc_anat = create_bids_structure(subj=subj, sess=sess, datatype='anat', root=data_path, 
                                                   folderlevel='derivatives', workingdir=cfg['prep_foldername'])   
                        atlas_labels = prepare_atlas_labels(cfg['atlas'], glob.glob(os.path.join(bids_strc_anat.get_path(), '*label*'))[0])
                else:
                    atlas_labels=''
                    
                # Define TPMs
                if cfg['atlas_TPM']:
                    bids_strc_reg_TPM  = create_bids_structure(subj=subj, sess=sess, datatype='registration', description=cfg['atlas_TPM']+'-To-'+'allDelta-allb', root=cfg['data_path'] , 
                                                 folderlevel='derivatives', workingdir=cfg['analysis_foldername'])
                    bids_strc_reg_TPM.set_param(base_name='')
                    TPMs = []
                    for tissue in ['GM', 'WM', 'CSF']:
                        path = bids_strc_reg_TPM.get_path(f'atlas_TPM_{tissue}_in_dwi.nii.gz')
                        TPMs.append(path if os.path.exists(path) else '')
                else:
                    TPMs = []
                    
                # If there is mrs voxel, add it as ROI
                bids_mrs  = create_bids_structure(subj=subj, sess=sess, datatype='registration', description=f"dmrs-to-allDelta-allb", root=data_path, 
                                               folderlevel='derivatives', workingdir=cfg['analysis_foldername'])
                bids_mrs.set_param(base_name='')
                if cfg['mrs_vx'] == 1 and os.path.exists(bids_mrs.get_path()):
                    ROI_list.append('voxel_mrs'); 
                    if TPMs:
                        ROI_list.append('voxel_mrs_GM')
                
                # If there is manual masks
                bids_manual = create_bids_structure(subj=subj, sess=sess, datatype='', root=data_path,
                     folderlevel='derivatives', workingdir='manual_masks')
                bids_manual.set_param(description='')
                
                # Determine output file
                bids_STE.set_param(datatype='dwi')
                outfile = os.path.join(os.path.dirname(os.path.join(bids_STE.get_path())), f"output_ROIs_{cfg['atlas']}_Micro_FA.xlsx")
                outfile2 = os.path.join(os.path.dirname(os.path.join(bids_STE.get_path())), f"output_ROIs_{cfg['atlas']}_Micro_FA.npy")

                # Option 1. Extract estimates with user defined ROIs
                patterns, lims, maximums = get_param_names_model('Micro_FA',cfg['is_alive'])
                cleaned_patterns = [re.sub(r'\*{2,}', '*', re.sub('Micro_FA', '', p, flags=re.IGNORECASE).replace('[^s]', '')).strip('*') for p in patterns]          
                
                if os.path.exists(atlas) and os.path.exists(output_path):
                    
                    # Get values of parameters inside the ROIs        
                    Data, Data_all, Data_l, Data_r = get_values_within_ROI(
                        ROI_list, atlas, atlas_labels, TPMs, cfg['tpm_thr'], 
                        vx_middle, patterns, maximums, bids_strc_reg, bids_mrs, bids_manual, output_path)
                    
                    # Save summary of means in excel format
                    df_data = []
                    df_data = pd.DataFrame(Data, columns=cleaned_patterns)
                    df_data.insert(0, 'ROI Name', ROI_list)
                    df_data.to_excel(outfile, index=False)
                    
                    # Save all data in npy format
                    df_data_all = pd.DataFrame(Data_all, columns=cleaned_patterns)
                    df_data_all.insert(0, 'ROI Name', ROI_list)
                    np.save(outfile2, df_data_all)
                    
                    if cfg['lat_ROIS']==1:
                        df_data_l = pd.DataFrame(Data_l, columns=cleaned_patterns)
                        df_data_l.insert(0, 'ROI Name', ROI_list)
                        np.save(outfile2.replace('.npy','_left.npy'), df_data_l)
                                
                        df_data_r = pd.DataFrame(Data_r, columns=cleaned_patterns)
                        df_data_r.insert(0, 'ROI Name', ROI_list)
                        np.save(outfile2.replace('.npy','_right.npy'), df_data_r)
                
            ######## MAKE ROI MAP ########
            if cfg['ROIs_GM'] + cfg['ROIs_WM']:
                bids_strc_reg  = create_bids_structure(subj=subj, sess=sess, datatype='registration', description=cfg['atlas']+'-To-'+'allDelta-allb', root=data_path, 
                                              folderlevel='derivatives', workingdir=cfg['analysis_foldername'])
                bids_strc_reg.set_param(base_name='')
                
                roi_paths =  []
                for ROI in cfg['ROIs_GM'] + cfg['ROIs_WM']:
                    roi_paths.append(bids_strc_reg.get_path(f'mask_{ROI}.nii.gz'))
               
                QA_ROIs(roi_paths, bids_strc_reg.get_path('ref_dwi.nii.gz'), bids_strc_reg.get_path(),cfg)
            
            if  cfg['ROIs_manual']:
                bids_manual = create_bids_structure(subj=subj, sess=sess, datatype='', root=data_path,
                     folderlevel='derivatives', workingdir='manual_masks')
                bids_manual.set_param(description='')
                
                roi_paths =  []
                for ROI in  cfg['ROIs_manual']:
                    roi_paths.append(bids_manual.get_path(f'mask_{ROI}.nii.gz'))
                    
                bids_strc = create_bids_structure(subj=subj, sess=sess, datatype="dwi", root=data_path, 
                                                folderlevel='derivatives', workingdir=cfg['prep_foldername'],description='allDelta-allb')
       
                QA_ROIs(roi_paths, bids_strc.get_path('b0_dn_gc_ec_avg_bc_brain.nii.gz'), bids_manual.get_path(),cfg)