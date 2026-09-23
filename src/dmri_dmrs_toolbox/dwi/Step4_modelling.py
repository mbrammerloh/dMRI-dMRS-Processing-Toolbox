"""
Script to analyse GM microstructure by fitting models such as Nexi, Sandi, ...
It uses the SwissKnife module installed in the SwissKnife python environment.

Last changed Jan 2025
@author: Rita O
"""
import os
import pandas as pd
import numpy as np
import glob
import fnmatch
import math
from dmri_dmrs_toolbox.misc.bids_structure import create_bids_structure
from dmri_dmrs_toolbox.misc.custom_functions import (create_directory, get_file_in_folder, copy_files_BIDS,
    multiply_by_mask, plot_summary_params_model, register_outputfits_to_anat,calculate_pwd_avg, compute_micro_FA,)
from dmri_dmrs_toolbox.misc.custom_functions_models import (get_param_names_model,run_swissknife_model, run_sandi_amico,run_sandi_mp_model, run_designer_model,
run_sandix_sj_model, prepare_model_inputs, run_dtidki_designer, get_data_used, run_uGUIDE_preparation, run_uGUIDE_model, run_xgboost_preparation)
from dmri_dmrs_toolbox.misc.atlas_functions import prepare_atlas_labels, create_ROI_mask


def Step4_modelling(cfg):
    
    data_path   = cfg['data_path']     
    scan_list   = pd.read_excel(os.path.join(data_path, cfg['scan_list_name'] ))
    cfg['model_list'] = cfg['model_list_GM'] + cfg['model_list_WM']
    # Define path to docker
    docker_path = '/data' 
    
    # Run preparation for uGUIDE (assumes NEXI has been run before)
    if any('uGUIDE' in model for model in cfg['model_list_GM']):
        cfg_uGUIDE = {}
        cfg_uGUIDE['model']="Nexi"
        cfg_uGUIDE['noise']="rician"
        cfg_uGUIDE['hidden_layers']=[50, 30]
        cfg_uGUIDE['nb_simu']=1_000_000 #700_000
        cfg_uGUIDE['nb_theta']=1_000
        run_uGUIDE_preparation(data_path, cfg, cfg_uGUIDE, scan_list)
        
    if any('xgboost' in model for model in cfg['model_list_GM']):
        cfg_xgboost = {}
        cfg_xgboost['model']="Nexi"
        cfg_xgboost['n_training_samples']= 1000000
        run_xgboost_preparation(data_path, cfg, cfg_xgboost, scan_list)

    ######## SUBJECT-WISE OPERATIONS ########
    for subj in cfg['subj_list']:
        
        print('Modelling ' + subj + '...')
    
        # Extract data for subject
        subj_data      = scan_list[(scan_list['study_name'] == subj)].reset_index(drop=True)
        subj_data      = subj_data[subj_data['analyse'] == 'y']
        
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
            
            ########################## DELTA-WISE OPERATIONS ##########################      
            filtered_data = subj_data_sess[(subj_data_sess['phaseDir'] == 'fwd') & (subj_data_sess['sessNo'] == sess) & (subj_data_sess['noBval'] > 1) & (subj_data_sess['acqType'] == 'PGSE')]
            Delta_list = filtered_data['diffTime'].unique().astype(int).tolist()
            
            for Delta in Delta_list:

                # Define bids structure for the processed data
                bids_strc_prep = create_bids_structure(subj=subj, sess=sess, datatype='dwi', root=data_path, 
                                                folderlevel='derivatives', workingdir=cfg['prep_foldername'])
                #bids_strc_prep.set_param(description='Delta_' + str(Delta) + '_fwd') # repricated: done on each Delta processed separately
                bids_strc_prep.set_param(description='allDelta-allb/Delta_' + str(Delta)) # new: done on each Delta processed together ('allDelta')

                ######## Run DTI and DKI ########  
                 
                # Define BIDS structure for the analysis data
                bids_strc_analysis = create_bids_structure(subj=subj, sess=sess, datatype="dwi", description=f'DTI_DKI_Delta_{Delta}', root=data_path, 
                                            folderlevel='derivatives', workingdir=cfg['analysis_foldername'])
                # Make output folder 
                output_path = bids_strc_analysis.get_path()
                
                # Just run model if it doesn't exist on the folder yet
                if not os.path.exists(output_path) or cfg['redo_modelling']:
                    run_dtidki_designer(
                             cfg,
                             bids_strc_prep,
                             output_path,
                             data_path,
                             docker_path,
                         )
                       
                if os.path.exists(output_path):
                    # Mask output with brain mask for better visualization
                    for filename in os.listdir(output_path):
                        if filename.endswith(".nii"):
                            multiply_by_mask(os.path.join(output_path, filename), # filename input
                                             os.path.join(output_path,'Output_masked'), # output folder
                                                     os.path.join(output_path,'inputs', 'mask.nii.gz'),cfg) # mask
                            
                    # Plot summary in dwi space
                    bids_strc_prep.set_param(description='allDelta-allb') # new: done on each Delta processed together ('allDelta')
                    plot_summary_params_model(os.path.join(output_path,'Output_masked'), 'DTI_DKI', cfg,bids_strc_prep.get_path('b0_dn_gc_ec_avg_bc_brain.nii.gz'))
                    
                    # Register to anat space
                    bids_strc_anat = create_bids_structure(subj=subj, sess=sess, datatype="anat", root=data_path, 
                                              folderlevel='derivatives', workingdir=cfg['prep_foldername'])
                    register_outputfits_to_anat(os.path.join(output_path,'Output_masked'),
                                                os.path.join(output_path,'Output_in_anat'),
                                            'DTI_DKI',cfg, bids_strc_anat, bids_strc_prep)
                
                    # Plot summary plot in anat space
                    bids_strc_reg  = create_bids_structure(subj=subj, sess=sess, datatype='registration', description=cfg['atlas']  +'-To-'+cfg['anat_format'], root=data_path, 
                                                folderlevel='derivatives', workingdir=cfg['analysis_foldername'])
                    bids_strc_reg.set_param(base_name='')
                    if cfg['subject_type']=='organoid' and os.path.exists(bids_strc_reg.get_path('mask_organoids.nii.gz')):
                             atlas=bids_strc_reg.get_path(f"atlas_in_{cfg['anat_format']}.nii.gz")
                             atlas_labels = prepare_atlas_labels(cfg['atlas'], glob.glob(os.path.join(bids_strc_anat.get_path(), '*label*'))[0])
                             mask = create_ROI_mask(atlas, atlas_labels, [], 'organoids', cfg['tpm_thr'], bids_strc_reg)
                             plot_summary_params_model(os.path.join(output_path,'Output_in_anat'), 'DTI_DKI', cfg, bids_strc_anat.get_path(f"{cfg['anat_format']}_bc_brain.nii.gz"), bids_strc_reg.get_path('mask_organoids.nii.gz'))
    
                    else:
                        plot_summary_params_model(os.path.join(output_path,'Output_in_anat'), 'DTI_DKI', cfg, bids_strc_anat.get_path(f"{cfg['anat_format']}_bc_brain.nii.gz"))

                
                ######## Compute PWD for LTE data ######## 
                
                # Create BIDS structures for LTE
                bids_LTE_temp = create_bids_structure(subj=subj, sess=sess, datatype='dwi', root=cfg['data_path'] , 
                             folderlevel='derivatives', workingdir=cfg['prep_foldername'],description=f'allDelta-allb/Delta_{Delta}')
                bids_LTE      = create_bids_structure(subj=subj, sess=sess, datatype='dwi', root=cfg['data_path'] , 
                             folderlevel='derivatives', workingdir=cfg['analysis_foldername'],description=f'pwd_avg_Delta_{Delta}')
              
                # Create pwd average of LTE, just if it doesn't exist yet
                if not os.path.exists(bids_LTE.get_path()) or cfg['redo_modelling']:
                    create_directory(bids_LTE.get_path())
                    calculate_pwd_avg(get_file_in_folder(bids_LTE_temp,'*dwi_dn_gc_ec.nii.gz'),
                                      get_file_in_folder(bids_LTE_temp,'*bvalsNom.txt'),
                                      get_file_in_folder(bids_LTE_temp,'*bvalsEff.txt'),
                                      bids_LTE.get_path(),
                                      np.nan)
             

            ######## Compute PWD for STE data (if exists) ######## 
            
            # Create BIDS structures for STE
            bids_STE_temp = create_bids_structure(subj=subj, sess=sess, datatype='dwi_STE', root=cfg['data_path'] , 
                          folderlevel='derivatives', workingdir=cfg['prep_foldername'],description='STE_fwd')
            bids_STE      = create_bids_structure(subj=subj, sess=sess, datatype='dwi_STE', root=cfg['data_path'] , 
                          folderlevel='derivatives', workingdir=cfg['analysis_foldername'],description='pwd_avg')
            
            # Create pwd average of STE, just if it doesn't exist yet
            if os.path.exists(bids_STE_temp.get_path('dwi_dn_gc_topup.nii.gz')):                
                 
                  if not os.path.exists(bids_STE.get_path()) or cfg['redo_modelling']:
                      create_directory(bids_STE.get_path())
                      calculate_pwd_avg(bids_STE_temp.get_path('dwi_dn_gc_topup.nii.gz'),
                                        bids_STE_temp.get_path('bvalsNom.txt'),
                                        bids_STE_temp.get_path('bvalsEff.txt'),
                                        bids_STE.get_path(),
                                        np.nan)
                      
            ######## Compute PWD for STE data registered to LTE (if exists) ######## 
             
            # Create BIDS structures for STE
            bids_STE_temp = create_bids_structure(subj=subj, sess=sess, datatype='dwi_STE', root=cfg['data_path'] , 
                           folderlevel='derivatives', workingdir=cfg['prep_foldername'],description='STE_fwd')
            bids_STE      = create_bids_structure(subj=subj, sess=sess, datatype='dwi_STE', root=cfg['data_path'] , 
                           folderlevel='derivatives', workingdir=cfg['analysis_foldername'],description='pwd_avg_in_LTE')
            bids_STE_reg      = create_bids_structure(subj=subj, sess=sess, datatype='registration', root=cfg['data_path'] , 
                          folderlevel='derivatives', workingdir=cfg['analysis_foldername'],description='STE-To-LTE_'+ "allDelta-allb")
            bids_STE_reg.set_param(base_name='')
            
            # Create pwd average of STE, just if it doesn't exist yet
            if get_file_in_folder(bids_STE_reg,'*dn_gc_topup.nii.gz'):                
                  
                   if not os.path.exists(bids_STE.get_path()) or cfg['redo_modelling']:
                       create_directory(bids_STE.get_path())
                       calculate_pwd_avg(get_file_in_folder(bids_STE_reg,'*dn_gc_topup.nii.gz'),
                                         bids_STE_temp.get_path('bvalsNom.txt'),
                                         bids_STE_temp.get_path('bvalsEff.txt'),
                                         bids_STE.get_path(),
                                         np.nan)
                          
            
            ######## Compute MicroFA if data exists ########  
            if get_file_in_folder(bids_STE_reg,'*dn_gc_topup.nii.gz'):   
               
                # 2. Computing microFA in python
                    # Define BIDS structure
                bids_LTE      = create_bids_structure(subj=subj, sess=sess, datatype='dwi', root=cfg['data_path'] , 
                             folderlevel='derivatives', workingdir=cfg['analysis_foldername'],description=f"pwd_avg_Delta_{cfg['LTEDelta_for_microFA']}")
                bids_STE      = create_bids_structure(subj=subj, sess=sess, datatype='dwi_STE', root=cfg['data_path'] , 
                               folderlevel='derivatives', workingdir=cfg['analysis_foldername'],description='pwd_avg_in_LTE')
                bids_LTE_aux  = create_bids_structure(subj=subj, sess=sess, datatype='dwi', root=cfg['data_path'] , 
                             folderlevel='derivatives', workingdir=cfg['prep_foldername'],description=f"allDelta-allb/Delta_{cfg['LTEDelta_for_microFA']}")
                mask          =  get_file_in_folder(bids_LTE_aux,'*mask.nii.gz')
                bids_STE_out      = create_bids_structure(subj=subj, sess=sess, datatype='dwi_STE', root=cfg['data_path'] , 
                              folderlevel='derivatives', workingdir=cfg['analysis_foldername'],description='microFA')
                output_path = bids_STE_out.get_path()
                
                # Just run model if it doesn't exist on the folder yet
                if not os.path.exists(output_path) or cfg['redo_modelling']:
                    compute_micro_FA(bids_LTE, bids_STE, mask, output_path)
                       
                if os.path.exists(output_path):
                    create_directory((os.path.join(output_path,'Output_masked')))
                    # Mask output with brain mask for better visualization
                    for filename in os.listdir(output_path):
                        if filename.endswith(".nii.gz"):
                            multiply_by_mask(os.path.join(output_path, filename), # filename input
                                             os.path.join(output_path,'Output_masked'), # output folder
                                                     mask,cfg) # mask
                            
                    # Plot summary plot in dwi space
                    bids_strc_prep.set_param(description='allDelta-allb') # new: done on each Delta processed together ('allDelta')
                    plot_summary_params_model(os.path.join(output_path, 'Output_masked'),'Micro_FA',cfg, bids_strc_prep.get_path('b0_dn_gc_ec_avg_bc_brain.nii.gz'))
       
            ########################## MODEL-WISE OPERATIONS ##########################       
            for model in cfg['model_list']:
                
                print('Working with ' + model + '...')
                
                # Define data to be used
                data_used = get_data_used(model, subj_data_sess, sess)
                print(f'Using data: {data_used}')
                
                # Define bids structure 
                bids_strc_analysis = create_bids_structure(subj=subj, sess=sess, datatype='dwi', root=data_path, 
                                            folderlevel='derivatives', workingdir=cfg['analysis_foldername'],description=model)
                bids_strc_prep = create_bids_structure(subj=subj, sess=sess, datatype="dwi", description=data_used, root=data_path, 
                                          folderlevel='derivatives', workingdir=cfg['prep_foldername'])
                
                # Define paths   
                output_path = bids_strc_analysis.get_path()
                input_path = os.path.join(output_path,'inputs')

                # Just run model if it doesn't exist on the folder yet
                if not os.path.exists(output_path) or cfg["redo_modelling"]:
                           
                    # Copy necessary files for analysis 
                    inputs = prepare_model_inputs(
                        model, bids_strc_prep, input_path,
                        subj, sess, cfg, data_path, docker_path
                    )
                
                    if model in ["Nexi", "Sandi", "Smex", "Sandix", "Nexi_wSTE", "Sandi_wSTE","Nexi_xgboost","Nexi_xgboost_powered_nls"]:
                        xgboost_model_path = os.path.join(data_path,'derivatives',cfg['analysis_foldername'],'xgboost_config','xgboost_train_model.json')
                        run_swissknife_model(model, inputs, output_path, subj, sess, cfg, xgboost_model_path)
                
                    elif model == "Sandi_amico":
                        run_sandi_amico(model, inputs, bids_strc_prep, input_path, output_path, cfg, filtered_data)
                
                    elif model in ["SMI", "SMI_wSTE"]:
                        run_designer_model(model, inputs, output_path, cfg,data_path, docker_path)
                
                    elif "Sandi_MP" in model:
                        run_sandi_mp_model(model, inputs, bids_strc_prep, input_path,output_path, subj, sess, cfg)
                
                    elif "Sandix_SJ" in model:
                        run_sandix_sj_model(model, inputs, input_path, output_path, subj, sess, cfg, data_path)
                        
                    elif "Nexi_uGUIDE" in model:
                        # needs NEXI to be run first to do the powder average signal for fitting
                        bids_strc_analysis = create_bids_structure(subj=subj, sess=sess, datatype='dwi', root=data_path, 
                                                    folderlevel='derivatives', workingdir=cfg['analysis_foldername'],description='Nexi')
                        inputs['pwd_dwi']=copy_files_BIDS(bids_strc_analysis, input_path, "powderaverage_dwi.nii.gz")
                        run_uGUIDE_model(model, inputs, data_path, subj, sess, cfg, cfg_uGUIDE)

         
                # Mask output for better visualization
                patterns, lims, maximums = get_param_names_model(model,cfg['is_alive'])
                for filename in os.listdir(output_path):
                    if any(fnmatch.fnmatch(filename, pattern) for pattern in patterns):
                        multiply_by_mask(os.path.join(output_path, filename), # filename input
                                         os.path.join(output_path,'Output_masked'), # output folder
                                         os.path.join(output_path,'inputs', 'mask_dil.nii.gz'),cfg) # mask
                # Plot summary plot in dwi space
                bids_strc_prep.set_param(description='allDelta-allb') # new: done on each Delta processed together ('allDelta')
                plot_summary_params_model(os.path.join(output_path,'Output_masked'), model, cfg, bids_strc_prep.get_path('b0_dn_gc_ec_avg_bc_brain.nii.gz'))
                
                # Register to anat space
                bids_strc_anat = create_bids_structure(subj=subj, sess=sess, datatype="anat", root=data_path, 
                                          folderlevel='derivatives', workingdir=cfg['prep_foldername'])                
                register_outputfits_to_anat(os.path.join(output_path,'Output_masked'),
                                            os.path.join(output_path,'Output_in_anat'),
                                            model,cfg, bids_strc_anat, bids_strc_prep)
                
                # Plot summary plot in anat space
                if cfg['subject_type']=='organoid':
                         bids_strc_reg  = create_bids_structure(subj=subj, sess=sess, datatype='registration', description=cfg['atlas']  +'-To-'+cfg['anat_format'], root=data_path, 
                                                     folderlevel='derivatives', workingdir=cfg['analysis_foldername'])
                         bids_strc_reg.set_param(base_name='')
                         atlas=bids_strc_reg.get_path(f"atlas_in_{cfg['anat_format']}.nii.gz")
                         atlas_labels = prepare_atlas_labels(cfg['atlas'], glob.glob(os.path.join(bids_strc_anat.get_path(), '*label*'))[0])
                         mask = create_ROI_mask(atlas, atlas_labels, [], 'organoids', cfg['tpm_thr'], bids_strc_reg)
                         plot_summary_params_model(os.path.join(output_path,'Output_in_anat'), model, cfg, bids_strc_anat.get_path(f"{cfg['anat_format']}_bc_brain.nii.gz"), bids_strc_reg.get_path('mask_organoids.nii.gz'))

                else:
                    plot_summary_params_model(os.path.join(output_path,'Output_in_anat'), model, cfg, bids_strc_anat.get_path(f"{cfg['anat_format']}_bc_brain.nii.gz"))

           
            
            


