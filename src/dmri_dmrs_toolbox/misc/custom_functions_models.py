#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 09:46:56 2026

@author: localadmin
"""
import os
import sys
import numpy as np
import nibabel as nib
import pandas as pd
import subprocess
from importlib.resources import files
import glob as glob
from pathlib import Path
import shutil
import json
from .bids_structure import create_bids_structure
from .custom_functions import (create_directory, modify_units_bvals, read_numeric_txt, binary_op, remove_folder, copy_files_BIDS, get_file_in_folder, extract_vols)

# ============================================================
# Helper functions
# ============================================================

def linear_model(b, m, b_int):
    return m * b + b_int

def get_param_names_model(model, is_alive):
    
    if model != 'DTI_DKI' and  model != 'Micro_FA':
        model  = model.split('_')[0]
    
    if model=='Nexi':
        if is_alive=='ex_vivo':
            patterns = ["*nexi*t_ex*", "*nexi*di*","*nexi*de*","*nexi*f*"]
            lims     = [(0, 150), (0, 2), (0, 2),  (0, 0.9)]
            maximums = np.array([[1, 150], (0.0, 2), (0.0, 2), [0, 0.9]])
        else:
            patterns = ["*nexi*t_ex*", "*nexi*di*","*nexi*de*","*nexi*f*"]
            lims     = [(0, 100), (0, 3.5), (0, 3.5),  (0, 1)]
            maximums = np.array([[1, 150], [0.1, 3.5], [0.1, 3.5], [0.1, 0.9]])
    
    elif model=='Smex':
        if is_alive=='ex_vivo':
            patterns = ["*smex*t_ex*", "*smex*di*","*smex*de*","*smex*f*"]
            lims     = [(0, 50), (0, 2), (0, 2),  (0, 0.4)]
            maximums = np.array([[1, 80], [0.1, 2], [0, 2], [0.1, 0.9]])
        else:
            patterns = ["*smex*t_ex*", "*smex*di*","*smex*de*","*smex*f*"]
            lims     = [(0, 80), (0, 3.5), (0, 2),  (0, 0.85)]
            maximums = np.array([[1, 80], [0.1, 3.5], [0.1, 3.5], [0.1, 0.9]])
    
    elif model=='Sandi':
        if is_alive=='ex_vivo':
            patterns = ["*sandi*di*","*sandi*de*","*sandi*fneurite*","*sandi*fsoma*","*sandi*rs*"]
            lims = [(0, 2), (0, 2),  (0, 0.9), (0,0.3),(0, 25)]
            maximums = np.full((len(patterns), 2), np.inf)
            maximums[:, 0] = -np.inf  
        else:
            patterns = ["*sandi*di*","*sandi*de*","*sandi*fneurite*", "*sandi*fsoma*","*sandi*rs*"]
            lims = [ (0, 3.5), (0, 3.5),  (0, 1), (0,1), (0, 20)]
            maximums = np.full((len(patterns), 2), np.inf)
            maximums[:, 0] = -np.inf  
        
    elif model=='Sandix':
        if is_alive=='ex_vivo':
             patterns = ["*sandix*t_ex*", "*sandix*di*","*sandix*de*","*sandix*fneurite*","*sandix*fsoma*","*sandix*rs*"]
             lims     = [(0, 50), (0, 2), (0, 2),  (0, 0.4), (0,0.3), (0, 25)]
             maximums = np.full((len(patterns), 2), np.inf)
             maximums[:, 0] = -np.inf 
        else:
            patterns = ["*sandix*t_ex*", "*sandix*di*","*sandix*de*","*sandix*fneurite*","*sandix*fsoma*","*sandix*rs*"]
            lims     = [(0, 100), (0, 3.5), (0, 3.5),  (0, 9), (0,0.3), (0, 25)]
            maximums = np.full((len(patterns), 2), np.inf)
            maximums[:, 0] = -np.inf 
        
    elif model=='SMI':
        patterns = ["*Da*", "*DePar*", "*DePerp*", "*f*", "*fw*", "*p2*", "*p4*"]
        lims = [(0, 4), (0, 4), (0, 4),  (0, 0.85), (0, 3), (0, 0.5), (0,0.5)]
        maximums = np.full((len(patterns), 2), np.inf)
        maximums[:, 0] = -np.inf  

    elif model=='DTI_DKI':
        if is_alive=='ex_vivo':
            patterns = ['*md_dki*','*mk_dki*','*fa_dki*']
            lims = [(0, 1), (0, 2), (0, 1)]
            #maximums = np.full((len(patterns), 2), np.inf)
            #maximums[:, 0] = -np.inf 
            maximums = np.array([[0, 5], [0, 50], [0, 50]])
        else:
            patterns = ['*md_dki*','*mk_dki*','*fa_dki*']
            lims = [(0, 2), (0, 2), (0, 1)]
            #maximums = np.full((len(patterns), 2), np.inf)
            #maximums[:, 0] = -np.inf 
            maximums = np.array([[0, 3], [0, 3], [0, 1]])
            
    elif model=='Micro_FA':
            patterns = ["*microFA*", "*MD*", "*MKiso*", "*MKaniso*", "*Uiso*", "*Uaniso*"]
            lims = [(0, 1), (0, 3),(0, 3),(0, 3),(0, 2),(0, 3)]
            maximums = np.array([[0, 1], [0, 3],[0, 3],[0, 3],[0, 2],[0, 3]])

    return patterns, lims, maximums

# def run_script_in_conda_environment(script_path,env_name):
#     subprocess.run(f"""conda init
#         source ~/.bashrc
#         source activate base
#         conda activate """+env_name+f"""
#         python """+script_path,
#             shell=True, executable='/bin/bash', check=True)

def compute_rmse(residual_file, output_file):
    nii = nib.load(residual_file)
    residuals = nii.get_fdata()
    rmse_map = np.sqrt(np.mean(residuals**2, axis=-1))
    rmse_nii = nib.Nifti1Image(rmse_map, nii.affine, nii.header)
    nib.save(rmse_nii, output_file)

def get_data_used(model, subj_data, sess):
    if any(m in model for m in ["Nexi", "Smex", "Sandix"]):
        return "allDelta-allb"
    
    filtered = subj_data[
        (subj_data["acqType"] == "PGSE") &
        (subj_data["phaseDir"] == "fwd") &
        (subj_data["sessNo"] == sess) &
        (subj_data["noBval"] > 1)
    ]
    
    if "Sandi" in model:
        ind = filtered["diffTime"].idxmin()
    elif model in ["SMI", "SMI_wSTE"]:
        ind = filtered["diffTime"].idxmax()
    else:
        raise ValueError(f"Unknown model: {model}")
    
    return f"allDelta-allb/Delta_{int(filtered['diffTime'][ind])}"


def get_mrs_prior(model, subj, sess, cfg, output_path):
    if "mrs_informed" not in model:
        return 0, 0, False

    mrs_model = "sphere_stick_sandi"

    bids_mrs = create_bids_structure(
        subj=subj, sess=sess, datatype="dmrs",
        root=cfg["data_path"],
        folderlevel="derivatives",
        workingdir=cfg["analysis_foldername"],
        description=mrs_model,
    )

    output_path_mrs = os.path.join(bids_mrs.get_path(), "csvs")

    if not os.path.isdir(output_path_mrs):
        print(f"Skipping {model} for {subj} {sess} (no MRS prior)")
        return None, None, True

    prior_r, prior_r_sd = [], []

    for metab in ["NAA+NAAG", "Glu", "Ins"]:
        data = pd.read_csv(
            os.path.join(output_path_mrs, f"fit_parameters_{metab}_{mrs_model}.csv")
        )
        prior_r.append(data["r"].iloc[0])
        prior_r_sd.append(data["r"].iloc[1])

    prior_r = np.mean(prior_r)
    prior_r_sd = np.mean(prior_r_sd)

    with open(os.path.join(output_path, "radius_used.txt"), "w") as f:
        f.write(f"{prior_r} {prior_r_sd}\n")

    return prior_r, prior_r_sd, False

def _model_clean_name(model):
    """
    Return the base model name used by graymatter_swissknife.

    Examples
    --------
    Sandi_wSTE          -> Sandi
    Sandix_mrs_informed -> Sandix
    Nexi                -> Nexi
    """
    has_ste = "wSTE" in model
    has_mrs = "mrs_informed" in model
    has_xgboost = "xgboost" in model

    return model.split("_")[0] if (has_ste or has_mrs or has_xgboost) else model

# ============================================================
# Prepare inputs and organize outputs
# ============================================================

def prepare_sandi_mp_inputs(inputs, bvecs, subj):
    files_to_copy = {
        inputs["sigma"]: "noisemap.nii.gz",
        inputs["dwi"]: "dwi.nii.gz",
        inputs["mask"]: "mask.nii.gz",
        inputs["bvals"]: "dwi.bval",
        bvecs: "dwi.bvec",
    }

    num = subj.split("-")[-1]
    sub = f"sub-{num}"

    dst_dir = None

    for file, ending in files_to_copy.items():
        src = Path(file)
        ses = [p for p in src.parts if p.startswith("ses-")][0]

        new_name = f"{sub}_{ses}_acq-01_run-01_desc-preproc_{ending}"
        dst_dir = src.parent.parent / "derivatives" / "preprocessed" / sub / ses

        create_directory(dst_dir)
        shutil.copy2(src, dst_dir / new_name)

    return dst_dir, sub, ses
       
def prepare_model_inputs(model, bids_strc_prep, input_path, subj, sess, cfg, data_path, docker_path):
    create_directory(input_path)

    inputs = {
        "dwi": copy_files_BIDS(bids_strc_prep, input_path, "dwi_dn_gc_ec.nii.gz"),
        "big_delta": copy_files_BIDS(bids_strc_prep, input_path, "DiffTime.txt"),
        "small_delta": copy_files_BIDS(bids_strc_prep, input_path, "DiffDuration.txt"),
        "bvals": copy_files_BIDS(bids_strc_prep, input_path, "bvalsNom.txt"),
        "sigma": copy_files_BIDS(bids_strc_prep, input_path, "dwi_dn_sigma.nii.gz"),
        "mask": copy_files_BIDS(bids_strc_prep, input_path, "mask_dil.nii.gz"),
        "input_file": None,
        "others": "",
    }

    inputs["new_bvals"] = inputs["bvals"].replace(".txt", "_units.txt")
    modify_units_bvals(inputs["bvals"], inputs["new_bvals"])

    if any(m in model for m in ["Nexi", "Sandi", "Smex", "Sandix"]):
        bids_lowb = create_bids_structure(
            subj=subj, sess=sess, datatype="dwi",
            description="allDelta-lowb", root=data_path,
            folderlevel="derivatives", workingdir=cfg["prep_foldername"]
        )
        inputs["sigma"] = copy_files_BIDS(bids_lowb, input_path, "dwi_dn_sigma.nii.gz")

    elif model == "SMI":
        inputs["input_file"] = copy_files_BIDS(
            bids_strc_prep, input_path, "dwi_dn_gc_ec.mif"
        ).replace(data_path, docker_path)

    elif model == "SMI_wSTE":
        bids_ste = create_bids_structure(
            subj=subj, sess=sess, datatype="dwi_STE",
            description="STE_fwd", root=data_path,
            folderlevel="derivatives", workingdir=cfg["prep_foldername"]
        )

        ste = copy_files_BIDS(
            bids_ste, input_path, "dwi_dn_gc_topup.mif"
        ).replace(data_path, docker_path)

        lte = copy_files_BIDS(
            bids_strc_prep, input_path, "dwi_dn_gc_ec.mif"
        ).replace(data_path, docker_path)

        inputs["input_file"] = f"{lte},{ste}"
        inputs["others"] = "-echo_time 51,51 -bshape 1,0 -compartments EAS,IAS -debug"

    return inputs

def save_sandi_fraction_maps(out_path):
    """
    Save fneurite and fsoma maps from SwissKnife SANDI/SANDIX outputs.

    SwissKnife stores:
        f  = total restricted/intracellular fraction
        fs = soma fraction within f

    Therefore:
        fneurite = f * (1 - fs)
        fsoma    = f * fs
    """
    import nibabel as nib
    
    map_f = glob.glob(os.path.join(out_path, "*f.nii.gz"))
    base_name = os.path.basename(map_f[0]).split('f.nii.gz')[0]
    img_f  = nib.load(map_f[0])
    hdr = img_f.header.copy()
    hdr.set_data_dtype(np.float32)
   
    map_f = nib.load(map_f[0]).get_fdata()
    
    map_fs = glob.glob(os.path.join(out_path, "*fs.nii.gz"))
    map_fs = nib.load(map_fs[0]).get_fdata()
   
    f_neurite = map_f*(1-map_fs)
    f_soma = map_f*map_fs
   
    ni_neurite = nib.Nifti1Image(f_neurite.astype(np.float32), affine=img_f.affine, header=hdr)
    ni_soma    = nib.Nifti1Image(f_soma.astype(np.float32),    affine=img_f.affine, header=hdr)
   
    nib.save(ni_neurite, os.path.join(out_path, f"{base_name}fneurite.nii.gz"))
    nib.save(ni_soma, os.path.join(out_path, f"{base_name}fsoma.nii.gz"))
 
def save_sandi_fraction_maps_experimental(out_path):
    """
    Save fneurite and fsoma maps from SwissKnife SANDI/SANDIX outputs.

    SwissKnife stores:
        f  = total restricted/intracellular fraction
        fs = soma fraction within f

    Therefore:
        fneurite = f * (1 - fs)
        fsoma    = f * fs
    """
    import nibabel as nib
    
    th1_paths = glob.glob(os.path.join(out_path, "*_theta1.nii.gz"))
    th2_paths = glob.glob(os.path.join(out_path, "*_theta2.nii.gz"))
    assert len(th1_paths) == 1 and len(th2_paths) == 1, "Expected exactly one theta1-map and one theta2-map."
    
    th1_path, th2_path = th1_paths[0], th2_paths[0]
    # base name before "_theta1.nii.gz"
    base_name = os.path.basename(th1_path).split("_theta1.nii.gz")[0]
    
    img_th1 = nib.load(th1_path)
    hdr = img_th1.header.copy()
    hdr.set_data_dtype(np.float32)
    
    theta1 = img_th1.get_fdata()
    theta2 = nib.load(th2_path).get_fdata()
    
    # optional: clamp to [0, π/2] if your optimizer used those bounds
    theta1 = np.clip(theta1, 0.0, np.pi/2)
    theta2 = np.clip(theta2, 0.0, np.pi/2)
    
    # MATLAB-style stick-breaking (angles)
    c1, s1 = np.cos(theta1), np.sin(theta1)
    c2, s2 = np.cos(theta2), np.sin(theta2)
    
    f_neurite = c1**2
    f_soma    = (s1**2) * (c2**2)
    f_total   = f_neurite + f_soma
    f_extra   = 1.0 - f_total
    
    
    aff = img_th1.affine
    nib.save(nib.Nifti1Image(f_neurite.astype(np.float32), aff, hdr), os.path.join(out_path, f"{base_name}_fneurite.nii.gz"))
    nib.save(nib.Nifti1Image(f_soma.astype(np.float32),    aff, hdr), os.path.join(out_path, f"{base_name}_fsoma.nii.gz"))
    nib.save(nib.Nifti1Image(f_extra.astype(np.float32),   aff, hdr), os.path.join(out_path, f"{base_name}_fextra.nii.gz"))

def move_outputs(src_dir, dst_dir, pattern_map):
    for pattern, new_name in pattern_map.items():
        matches = glob.glob(str(os.path.join(src_dir, pattern)))

        if not matches:
            print(f"No file found for pattern: {pattern}")
            continue

        src = Path(matches[0])
        dst = Path(dst_dir) / new_name

        shutil.move(src, dst)
        print(f"Moved: {src} -> {dst}")


 # ============================================================
 # Run models
 # ============================================================  
     
def run_auxiliar_model(cfg, env_name, args):
    script_path = files("dmri_dmrs_toolbox.misc").joinpath("run_in_env_models.py")
    command = [
        cfg["conda_exe"], "run", "-n", env_name,
        "python", str(script_path)
    ] + args
    subprocess.run(command, check=True)

def run_matlab_command(matlab_cmd):
    cmd = [
        "matlab", "-nodisplay", "-nosplash", "-nodesktop",
        "-r", matlab_cmd,
    ]
    subprocess.run(cmd, check=True)
    
    
def run_swissknife_model(model, inputs, output_path, subj, sess, cfg, xgboost_modle_path):

     if model=='Nexi_xgboost':
        algo = 'xgboost'
     elif model=='Nexi_xgboost_powered_nls':
            algo = 'xgboost_powered_nls'
     else:
        algo = 'nls'
        
     model_clean = _model_clean_name(model)

     args = [
         model_clean,
         output_path,
         inputs["dwi"],
         inputs["new_bvals"],
         inputs["big_delta"],
         inputs["small_delta"],
         inputs["sigma"],
         inputs["mask"],
         cfg["is_alive"],
         algo,
         xgboost_modle_path,
         "--debug",
     ]
    
     env_name = "SwissKnife"
    
     if model == "Nexi_wSTE":
         bids_ste = create_bids_structure(
             subj=subj, sess=sess, datatype="dwi_STE",
             root=cfg["data_path"],
             folderlevel="derivatives",
             workingdir=cfg["analysis_foldername"],
             description="microFA",
         )
         uFA = os.path.join(bids_ste.get_path(), "Uaniso.nii")
         args.insert(-1, uFA)
         env_name = "SwissKnife_exp"
    
     elif model == "Sandi_wSTE":
         bids_ste = create_bids_structure(
             subj=subj, sess=sess, datatype="dwi_STE",
             root=cfg["data_path"],
             folderlevel="derivatives",
             workingdir=cfg["analysis_foldername"],
             description="pwd_avg_in_LTE",
         )
    
         ste_data = os.path.join(
             bids_ste.get_path(),
             "STE_in_LTE_dn_gc_topup_pwd_avg_norm.nii.gz",
         )
         ste_bvals = get_file_in_folder(bids_ste, "*STE_fwd_bvalsNom_avg.txt")
    
         args.insert(-1, ste_data)
         args.insert(-1, ste_bvals)
         env_name = "SwissKnife_exp"
             
     run_auxiliar_model(cfg, env_name, args)
    
     res = os.path.join(output_path, "mean_residuals.nii.gz")
     if os.path.exists(res):
         compute_rmse(res, os.path.join(output_path, "RMSE.nii.gz"))    
         
     if 'Sandi' in model or 'Sandix' in model:
         save_sandi_fraction_maps(output_path)

        
def run_xgboost_preparation(data_path, cfg, cfg_xgboost, scan_list):
    from warnings import warn
    
    main_folder = os.path.join(data_path,'derivatives',cfg['analysis_foldername'])
    model              = cfg_xgboost['model']
    n_training_samples = cfg_xgboost['n_training_samples']
    xgboost_model_path = os.path.join(main_folder,"xgboost_config")
    
    # Just run this step if it wasn't done before
    if not os.path.exists(xgboost_model_path):
        
        create_directory(xgboost_model_path)
        
        # Loop through subjects to get sigma map and mask
        sigma_files = []
        for subj in cfg['subj_list']:
        
            # Extract data for subject
            subj_data      = scan_list[(scan_list['study_name'] == subj)].reset_index(drop=True)
            subj_data      = subj_data[subj_data['analyse'] == 'y']

            for sess in list(subj_data['sessNo'].unique()) :
                bids_strc_nexi = create_bids_structure(subj=subj, sess=sess, datatype="dwi", description='Nexi', root=data_path, 
                                          folderlevel='derivatives', workingdir=cfg['analysis_foldername'])
                if not os.path.exists(bids_strc_nexi.get_path()):
                    warn(f'WARNING: Missing NEXI results. Aborting')
                    return
                else:
                    sigma_files.append(get_file_in_folder(bids_strc_nexi,'*powderaverage_signal.npz'))

            # Chose subject that has all the Deltas
            best_subj = None
            best_sess = None
            best_filtered_data = None
            max_n_deltas = 0
            for subj in cfg['subj_list']:
            
                subj_data = scan_list[scan_list['study_name'] == subj].reset_index(drop=True)
                subj_data = subj_data[subj_data['analyse'] == 'y']
            
                for sess in list(subj_data['sessNo'].unique()):
            
                    filtered_data_tmp = subj_data[
                        (subj_data['phaseDir'] == 'fwd') &
                        (subj_data['sessNo'] == sess) &
                        (subj_data['noBval'] > 1) &
                        (subj_data['acqType'] == 'PGSE')
                    ]
            
                    n_deltas = filtered_data_tmp['diffTime'].nunique()
            
                    if n_deltas > max_n_deltas:
                        max_n_deltas = n_deltas
                        best_subj = subj
                        best_sess = sess
                        best_filtered_data = filtered_data_tmp.copy()
            subj = best_subj
            sess = best_sess
            filtered_data = best_filtered_data
            
            # Get protocol info from data
            Delta_list = filtered_data['diffTime'].unique().astype(int).tolist()
            xgboost_bvals = []
            xgboost_nshells = []
            xgboost_Delta = []
            xgboost_small_delta = filtered_data['diffDuration'].unique().astype(float)[0]
            for Delta in Delta_list:
                bids_strc_prep = create_bids_structure(subj=subj, sess=sess, datatype="dwi", description=f'Delta_{Delta}_fwd', root=data_path, 
                                          folderlevel='derivatives', workingdir=cfg['prep_foldername'])
                bvals = read_numeric_txt(get_file_in_folder(bids_strc_prep,'*bvalsNom.txt'))[0]
                bvals, idx, counts = np.unique(bvals, return_index=True, return_counts=True)
                mask = bvals > 0
                bvals = bvals[mask]
                counts = counts[mask]
                xgboost_bvals.append(bvals)
                xgboost_nshells.append(counts)
                xgboost_Delta.extend([Delta] * len(bvals))
            xgboost_bvals = np.hstack(xgboost_bvals).ravel()
            if np.nanmax(xgboost_bvals) > 100:
                xgboost_bvals = xgboost_bvals / 1000
            xgboost_nshells = np.hstack(xgboost_nshells).ravel()
            xgboost_Delta = np.hstack(xgboost_Delta).ravel()
            
        # Run Simulation of data
        env_name = "SwissKnife"
        
        code = r"""
import sys
import json
import numpy as np
import os

# Import the XGBoost functions
from graymatter_swissknife.xgboost.define_xgboost_model import define_xgboost_model
from graymatter_swissknife.xgboost.apply_xgboost_model import apply_xgboost_model
from graymatter_swissknife.models.parameters.acq_parameters import AcquisitionParameters
from graymatter_swissknife.models.find_model import find_model

out_folder    = sys.argv[1]
base_model    = sys.argv[2]
b             = np.array(json.loads(sys.argv[3]), dtype=float)
delta         = np.array(json.loads(sys.argv[4]), dtype=float)
small_delta   = json.loads(sys.argv[5])
sigma_files   = json.loads(sys.argv[6])

sigma_all = []
for sigma_file in sigma_files:
    powder_average_signal_npz = np.load(sigma_file)
    sigma = powder_average_signal_npz['sigma']
    sigma_all.append(sigma.T.ravel())   
sigma_all = np.concatenate(sigma_all)

# No initialization in XGBoost
estimation_init = None

# Define the XGBoost model from a file or generate and train it
xgboost_model_path = os.path.join(out_folder,'xgboost_train_model.json')
microstruct_model = find_model(base_model + 'RiceMean')
#microstruct_model = find_model(base_model)

acq_param = AcquisitionParameters(b=b, delta=delta, small_delta=small_delta)
n_training_samples  = int(json.loads(sys.argv[7]))

# Check if the XGBoost model path is provided
assert xgboost_model_path is not None, "The XGBoost model path must be provided, either to save or load the model."

xgboost_model = define_xgboost_model(xgboost_model_path, False, microstruct_model,
                                        acq_param, n_training_samples, sigma=sigma_all, force_cpu=False, n_cores=-1)
  
"""
        
        print('Running training xgboost model...')
        subprocess.run(
            [
                cfg["conda_exe"], "run", "-n", env_name, "python", "-c", code,
                str(os.path.join(main_folder,"xgboost_config")),
                str(model),
                json.dumps(xgboost_bvals.tolist()),
                json.dumps(xgboost_Delta.tolist()),
                json.dumps(xgboost_small_delta),
                json.dumps(sigma_files),
                json.dumps(n_training_samples),        
            ],
            check=True,
        )
          

def run_uGUIDE_preparation(data_path, cfg, cfg_uGUIDE, scan_list):
    from warnings import warn
    import math

    main_folder = Path(os.path.join(data_path,'derivatives',cfg['analysis_foldername']))
    model           = cfg_uGUIDE['model']
    noise           = cfg_uGUIDE['noise']
    hidden_layers   = cfg_uGUIDE['hidden_layers']
    nb_simu         = cfg_uGUIDE['nb_simu']
    nb_theta        = cfg_uGUIDE['nb_theta']
    
    # Just run this step if it wasn't done before
    if not (main_folder / "uGUIDE_config").exists():
        
        # Loop through subjects to get sigma map and mask
        sigma_files = []
        mask_files = []
        for subj in cfg['subj_list']:
        
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

            for sess in sess_list:
                bids_strc_nexi = create_bids_structure(subj=subj, sess=sess, datatype="dwi", description='Nexi', root=data_path, 
                                          folderlevel='derivatives', workingdir=cfg['analysis_foldername'])
                if not os.path.exists(bids_strc_nexi.get_path()):
                    warn(f'WARNING: Missing NEXI results. Aborting')
                    return
                else:
                    sigma_files.append(get_file_in_folder(bids_strc_nexi,'*normalized_sigma.nii.gz'))
                    mask_files.append(get_file_in_folder(bids_strc_nexi,'*updated_mask.nii.gz'))
                   
        # Chose subject that has all the Deltas
        best_subj = None
        best_sess = None
        best_filtered_data = None
        max_n_deltas = 0
        for subj in cfg['subj_list']:
        
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
                 
            for sess in sess_list:
        
                filtered_data_tmp = subj_data[
                    (subj_data['phaseDir'] == 'fwd') &
                    (subj_data['sessNo'] == sess) &
                    (subj_data['noBval'] > 1) &
                    (subj_data['acqType'] == 'PGSE')
                ]
        
                n_deltas = filtered_data_tmp['diffTime'].nunique()
        
                if n_deltas > max_n_deltas:
                    max_n_deltas = n_deltas
                    best_subj = subj
                    best_sess = sess
                    best_filtered_data = filtered_data_tmp.copy()
        subj = best_subj
        sess = best_sess
        filtered_data = best_filtered_data
        
        # Get protocol info from data
        Delta_list = filtered_data['diffTime'].unique().astype(int).tolist()
        uGUIDE_bvals = []
        uGUIDE_nshells = []
        uGUIDE_Delta = []
        uGUIDE_small_delta = filtered_data['diffDuration'].unique().astype(float)[0]
        for Delta in Delta_list:
            bids_strc_prep = create_bids_structure(subj=subj, sess=sess, datatype="dwi", description=f'Delta_{Delta}_fwd', root=data_path, 
                                      folderlevel='derivatives', workingdir=cfg['prep_foldername'])
            bvals = read_numeric_txt(get_file_in_folder(bids_strc_prep,'*bvalsNom.txt'))[0]
            bvals, idx, counts = np.unique(bvals, return_index=True, return_counts=True)
            mask = bvals > 0
            bvals = bvals[mask]
            counts = counts[mask]
            uGUIDE_bvals.append(bvals)
            uGUIDE_nshells.append(counts)
            uGUIDE_Delta.extend([Delta] * len(bvals))
        uGUIDE_bvals = np.hstack(uGUIDE_bvals).ravel()
        if np.nanmax(uGUIDE_bvals) > 100:
            uGUIDE_bvals = uGUIDE_bvals / 1000
        uGUIDE_nshells = np.hstack(uGUIDE_nshells).ravel()
        uGUIDE_Delta = np.hstack(uGUIDE_Delta).ravel()

        _, adjust_parameter_limits, _ = get_param_names_model(model,cfg['is_alive'])
        
        # Run Simulation of data
        env_name = "SwissKnife"
        script_path = files("dmri_dmrs_toolbox.misc.uGUIDE").joinpath("uGUIDE_simulate_data.py")
        
        code = r"""
import sys
import json
import numpy as np
sys.path.insert(0, sys.argv[1])
from uGUIDE_simulate_data import simulate_data
from pathlib import Path

main_folder   = Path(sys.argv[2])
b             = np.array(json.loads(sys.argv[3]), dtype=float)
delta         = np.array(json.loads(sys.argv[4]), dtype=float)
nb_directions = np.array(json.loads(sys.argv[5]), dtype=int)
small_delta   = json.loads(sys.argv[6])
sigma_files   = json.loads(sys.argv[7])
mask_files    = json.loads(sys.argv[8])
lims          = np.array(json.loads(sys.argv[9]), dtype=float)
model         = json.loads(sys.argv[10])

simulate_data(main_folder, b, delta, nb_directions, small_delta, sigma_files, mask_files, [model], lims)
"""
        
        print('Running simulation of data for uGUIDE...')
        subprocess.run(
            [
                cfg["conda_exe"], "run", "-n", env_name, "python", "-c", code,
                str(script_path.parent),
                str(main_folder),
                json.dumps(uGUIDE_bvals.tolist()),
                json.dumps(uGUIDE_Delta.tolist()),
                json.dumps(uGUIDE_nshells.tolist()),
                json.dumps(uGUIDE_small_delta),
                json.dumps(sigma_files),
                json.dumps(mask_files),
                json.dumps(adjust_parameter_limits),
                json.dumps(model),
            ],
            check=True,
        )
        
        # run training of data
        env_name = "uGUIDE"
        script_path = files("dmri_dmrs_toolbox.misc.uGUIDE").joinpath("uGUIDE_training.py")
        

        code = r"""
import sys
import json
import numpy as np
sys.path.append(sys.argv[1])
from uGUIDE_training import model_training
from pathlib import Path

main_folder   = Path(sys.argv[2])
model         = sys.argv[3]
noise         = sys.argv[4]
hidden_layers = np.array(json.loads(sys.argv[5]), dtype=int)
nb_simu       = json.loads(sys.argv[6])
nb_theta      = json.loads(sys.argv[7])

model_training(main_folder, model, noise, hidden_layers, nb_simu, nb_theta)

"""
        
        print('Running uGUIDE training step...')
        subprocess.run(
            [
                cfg["conda_exe"], "run", "-n", env_name, "python", "-c", code,
                str(script_path.parent),
                str(main_folder),
                str(model),
                str(noise),
                json.dumps(hidden_layers),
                json.dumps(nb_simu),
                json.dumps(nb_theta),
            ],
            check=True
        )
        
 
def run_uGUIDE_model(model, inputs, data_path, subj, sess, cfg, cfg_uGUIDE):
    
    main_folder = Path(os.path.join(data_path,'derivatives',cfg['analysis_foldername']))
    model_clean  = model.split('_')[0] 
    
    model           = model_clean
    noise           = cfg_uGUIDE['noise']
    hidden_layers   = cfg_uGUIDE['hidden_layers']
    nb_simu         = cfg_uGUIDE['nb_simu']
    nb_theta        = cfg_uGUIDE['nb_theta']
    
    # run inference of data
    env_name = "uGUIDE"
    script_path = files("dmri_dmrs_toolbox.misc.uGUIDE").joinpath("uGUIDE_estimate_params_real_data.py")
    
    uguide_folder = main_folder / f"{subj}" /  f"ses-{sess:02}" / "dwi"  / f"{model}_uGUIDE"
    dwi_path  = inputs['pwd_dwi']
    mask_path = inputs['mask']


    code = r"""
import sys
import json
import numpy as np
sys.path.insert(0, sys.argv[1])
from uGUIDE_estimate_params_real_data import main_model_fit
from pathlib import Path

main_folder    = Path(sys.argv[2])
results_folder = Path(sys.argv[3])
dwi_path       = Path(sys.argv[4])
mask_path      = Path(sys.argv[5])
model          = sys.argv[6]
noise          = sys.argv[7]
hidden_layers = np.array(json.loads(sys.argv[8]), dtype=int)
nb_simu       = json.loads(sys.argv[9])
nb_theta      = json.loads(sys.argv[10])

main_model_fit(main_folder, results_folder, dwi_path, mask_path,
               model, noise, hidden_layers, nb_simu, nb_theta)
"""
        
    print('Running parameter estimation with uGUIDE...')
    subprocess.run(
        [
            cfg["conda_exe"], "run", "-n", env_name, "python", "-c", code,
            str(script_path.parent),
            str(main_folder),
            str(uguide_folder),
            str(dwi_path),
            str(mask_path),
            str(model),
            str(noise),
            json.dumps(hidden_layers),
            json.dumps(nb_simu),
            json.dumps(nb_theta),
        ],
        check=True
    )
    
    
    # Extract maps and build individial nifti
    MAP_estimates =  main_folder / f"{subj}" /  f"ses-{sess:02}" / "dwi" / f"{model}_uGUIDE" 

    pattern_map={   0: "nexi_uGUIDE_t_ex.nii.gz",
                    1: "nexi_uGUIDE_di.nii.gz",
                    2: "nexi_uGUIDE_de.nii.gz",
                    3: "nexi_uGUIDE_f.nii.gz",}

    for i in pattern_map:
        outputpath = MAP_estimates / pattern_map[i]
        extract_vols(
            str(MAP_estimates / "uGUIDE_MAP_Nexi.nii.gz"),    
            str(outputpath),
            i,
            1,
            cfg,
        )
           
def run_sandi_amico(model, inputs, bids_strc_prep, input_path, output_path, cfg, filtered_data):
    
     bvecs = copy_files_BIDS(bids_strc_prep, input_path, "bvecsRotated.txt")
     bvec_file = os.path.join(os.path.dirname(bvecs), "bvecs.bvec")
     bval_file = os.path.join(os.path.dirname(bvecs), "bvals.bval")
    
     shutil.copy2(bvecs, bvec_file)
     shutil.copy2(inputs["bvals"], bval_file)
    
     TE = np.unique(filtered_data["EPIfact"])[0]
    
     args = [
         model,
         output_path,
         inputs["dwi"],
         bval_file,
         inputs["big_delta"],
         inputs["small_delta"],
         inputs["sigma"],
         inputs["mask"],
         cfg["is_alive"],
         bvec_file,
         str(float(TE)),
         "--debug",
     ]
    
     run_auxiliar_model(cfg, "amico", args)
    
     move_outputs(
         src_dir=os.path.join(output_path, "AMICO", "SANDI"),
         dst_dir=output_path,
         pattern_map={
             "fit_Din.nii.gz": "sandi_di.nii.gz",
             "fit_De.nii.gz": "sandi_de.nii.gz",
             "fit_fneurite.nii.gz": "sandi_fneurite.nii.gz",
             "fit_fsoma.nii.gz": "sandi_fsoma.nii.gz",
             "fit_Rsoma.nii.gz": "sandi_rs.nii.gz",
             "fit_RMSE.nii.gz": "RMSE.nii.gz",
         },
     )

def run_designer_model(model, inputs, output_path, cfg,data_path, docker_path):
    args = [
        model,
        output_path.replace(data_path, docker_path),
        inputs["input_file"],
        inputs["mask"].replace(data_path, docker_path),
        inputs["sigma"].replace(data_path, docker_path),
        data_path,
        inputs["others"],
    ]
 
    run_auxiliar_model(cfg, "pipeline", args)

def run_sandi_mp_model(model, inputs, bids_strc_prep, input_path,output_path, subj, sess, cfg):
     
        prior_r, prior_r_sd, skip_model = get_mrs_prior(
            model, subj, sess, cfg, output_path)
     
        if skip_model:
            return
        
        bvecs = copy_files_BIDS(bids_strc_prep, input_path, "bvecsRotated.txt")
     
        dst_dir, sub, ses = prepare_sandi_mp_inputs(
            inputs, bvecs, subj
        )
     
        big_delta_val = float(np.loadtxt(inputs["big_delta"])[0])
        small_delta_val = float(np.loadtxt(inputs["small_delta"])[0])
     
        info_txt = dst_dir / "info.txt"
        with open(info_txt, "w") as f:
            f.write(f"{int('mrs_informed' in model)} {prior_r} {prior_r_sd}\n")
     
        sandi_folder = dst_dir.parents[3]
     
        matlab_cmd = (
            "try, "
            f"addpath(genpath('{os.path.join(cfg['toolboxes'], 'SANDI')}')); "
            f"SANDI_batch_analysis('{sandi_folder}', {big_delta_val}, {small_delta_val}, []); "
            "catch, exit(1), end, exit(0);"
        )
     
        run_matlab_command(matlab_cmd)
     
        sandi_output = (
            sandi_folder / "derivatives" / "SANDI_analysis" /
            sub / ses / "SANDI_Output"
        )
     
        residuals = sandi_output / "mean_residuals.nii.gz"
     
        binary_op(
            sandi_output / "diravg_signal.nii.gz",
            sandi_output / "SANDI-predicted_diravg.nii.gz",
            "-sub",
            residuals,
            cfg,
        )
     
        compute_rmse(residuals, sandi_output / "rmse_map.nii.gz")
     
        move_outputs(
            src_dir=sandi_output,
            dst_dir=sandi_folder,
            pattern_map={
                "SANDI-fit_Din.nii.gz": "sandi_di.nii.gz",
                "SANDI-fit_De.nii.gz": "sandi_de.nii.gz",
                "SANDI-fit_fneurite.nii.gz": "sandi_fneurite.nii.gz",
                "SANDI-fit_fsoma.nii.gz": "sandi_fsoma.nii.gz",
                "SANDI-fit_Rsoma.nii.gz": "sandi_rs.nii.gz",
                "mean_residuals.nii.gz": "mean_residuals.nii.gz",
                "rmse_map.nii.gz": "RMSE.nii.gz",
            },
        )
        
def run_sandix_sj_model(model, inputs, input_path, output_path, subj, sess, cfg, data_path):
   bids_mrs_mask = create_bids_structure(
                   subj=subj, sess=sess, datatype="registration",
                   description="dmrs-to-allDelta-allb",
                   root=data_path,
                   folderlevel="derivatives",
                   workingdir=cfg["analysis_foldername"])

   mrs_mask = copy_files_BIDS(
       bids_mrs_mask, input_path, "voxel_mrs.nii.gz"
   )

   prior_r, prior_r_sd, skip_model = get_mrs_prior(
       model, subj, sess, cfg, output_path
   )

   if skip_model:
       return

   matlab_cmd = (
       "try, "
       f"addpath(genpath('{os.path.join(cfg['toolboxes'], 'Sandix')}')); "
       f"addpath(genpath('{os.path.join(cfg['toolboxes'], 'spm12')}')); "
       f"SANDIX_analysis_RO("
       f"'{inputs['dwi']}', '{mrs_mask}', "
       f"'{inputs['big_delta']}', '{inputs['small_delta']}', "
       f"'{inputs['new_bvals']}', '{output_path}', "
       f"{prior_r}, {prior_r_sd}); "
       "catch, exit(1), end, exit(0);"
   )

   run_matlab_command(matlab_cmd)

def run_dtidki_designer(cfg, bids_strc_prep, output_path, data_path, docker_path):
    
    # Decide which model to run
    run_DTIDKI = 1
    bvals = np.unique(read_numeric_txt(get_file_in_folder(bids_strc_prep, '*bvalsNom.txt'))[0])
    has_low  = np.any(bvals <= 1000)
    has_high = np.any(bvals > 1000)
    
    do_model = []
    if has_high:  # at least one shell above 1000
        do_model.append('-DKI')
    if has_low:  # all bvals <= 1000
        do_model.append('-DTI')
    if has_high==0 and has_low==0:  # no valid shells found
        run_DTIDKI = 0
        print('No valid shells found for neither DTI nor DKI!')
    do_model = ' '.join(do_model)
    
    if run_DTIDKI == 1:
        if cfg["use_server_mount"] == 1:
            mount_path = os.path.join(os.path.expanduser("~"), "temp", "current_sub")
            input_path = os.path.join(mount_path, "inputs")
            remove_folder(mount_path)
            needs_copy_back = True
        else:
            mount_path = data_path
            input_path = os.path.join(output_path, "inputs")
            needs_copy_back = False
    
        create_directory(input_path)
    
        # --- Copy inputs
        dwi = copy_files_BIDS(bids_strc_prep, input_path, "dwi_dn_gc_ec.mif")
        mask = copy_files_BIDS(bids_strc_prep, input_path, "mask.nii.gz")
    
        # --- Convert to docker paths
        dwi = dwi.replace(mount_path, docker_path)
        mask = mask.replace(mount_path, docker_path)
    
        out_folder = (
            docker_path
            if cfg["use_server_mount"]
            else output_path.replace(data_path, docker_path)
        )
    
        extra = "-maxb 7" if cfg["is_alive"] == "ex_vivo" else ""
    
        command = (
            f"docker run -v {mount_path}:{docker_path} "
            f"nyudiffusionmri/designer2:v2.0.13 "
            f"tmi {do_model} {extra} "
            f"{dwi} {out_folder} -fit_constraints 0,1,0"
        )
    
        print(command)
        os.system(command)
    
        if needs_copy_back:
            shutil.copytree(mount_path, output_path, dirs_exist_ok=True)
            
def run_dtidki_dipy(cfg, bids_strc_prep, output_path, data_path, docker_path, do_model):
    # experimental
    from dipy.core.gradients import gradient_table
    import nibabel as nib
    import dipy.reconst.dti as dti
    import dipy.reconst.dki as dki

    # Load data
    out_path    = sys.argv[2]
    dwi_path    = sys.argv[3]  
    bvals_path  = sys.argv[4]
    bvecs_path  = sys.argv[5]
    mask_path   = sys.argv[6]
    bvecs=np.loadtxt(bvecs_path)
    bvals=np.loadtxt(bvals_path)
    img = nib.load(dwi_path)
    data = img.get_fdata()
    mask_data = nib.load(mask_path).get_fdata() > 0
            
    # Prepare gradients
    mask_b = bvals <= 3000
    gtab = gradient_table(bvals=bvals[mask_b], bvecs=bvecs[:,mask_b])
    
    # Fit
    print('Building DKI model...')
    dkimodel = dki.DiffusionKurtosisModel(gtab,fit_method='OLS')
    print('Fitting DKI...')
    dkifit = dkimodel.fit(data[:,:,:,mask_b],mask=mask_data)

    # Save maps
    MD = dkifit.md*1e3 # units um^2/ms
    MD_img = nib.Nifti1Image(MD.astype("float32"), affine=img.affine, header=img.header)
    nib.save(MD_img, os.path.join(out_path,'MD.nii'))
    
    MK = dkifit.mk()   # no units
    MK_img = nib.Nifti1Image(MK.astype("float32"), affine=img.affine, header=img.header)
    nib.save(MK_img, os.path.join(out_path,'MK.nii'))
    
    FA = dkifit.fa   # no units
    FA_img = nib.Nifti1Image(FA.astype("float32"), affine=img.affine, header=img.header)
    nib.save(FA_img, os.path.join(out_path,'FA.nii'))
    