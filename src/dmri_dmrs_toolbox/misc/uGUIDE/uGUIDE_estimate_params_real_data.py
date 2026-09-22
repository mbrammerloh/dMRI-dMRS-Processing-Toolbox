import os
import shlex
import subprocess
import time
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed
import nibabel as nib
import plotly.express as px
import plotly.graph_objects as go

from uGUIDE.config_utils import load_config_uGUIDE
from uGUIDE.estimation import estimate_microstructure


# ============================================================
# Helper functions
# ============================================================

def postprocess_NEXI_SANDIX(samples, config, adjust_parameter_limits=None):
    """
    Post-process uGUIDE samples:
    - convert U0/U1 into Di/De
    - for SANDIX, convert f/fs into neurite and soma volume fractions
    """
    prior_keys = np.array(list(config["prior"].keys()))

    idx_u0 = np.where(prior_keys == "U0")[0][0]
    idx_u1 = np.where(prior_keys == "U1")[0][0]

    u0 = samples[:, idx_u0]
    u1 = samples[:, idx_u1]

    u0 = np.clip(u0, 0, 1)
    u1 = np.clip(u1, 0, 1)

    Di_min, Di_max = config["prior_postprocessing"]["Di"]
    De_min, _ = config["prior_postprocessing"]["De"]
    
    # Di_min = 0.1
    # Di_max = 2 #3.5
    # De_min = 0.1

    Di = np.sqrt((Di_max - Di_min) ** 2 * u0) + Di_min
    De = (Di - Di_min) * u1 + De_min

    out_samples = samples.copy()
    out_samples[:, idx_u0] = Di
    out_samples[:, idx_u1] = De

    if config["microstructure_model_name"].lower() == "sandix":
        idx_f = np.where(prior_keys == "f")[0][0]
        idx_fs = np.where(prior_keys == "fs")[0][0]

        f = samples[:, idx_f]
        fs = samples[:, idx_fs]

        f_neurites = f * (1 - fs)
        f_soma = f * fs

        out_samples[:, idx_f] = f_neurites
        out_samples[:, idx_fs] = f_soma

    return out_samples



def get_slice_and_masks(res, mask_map, mask_degeneracy, slice_idx, ax_plot, param_idx):
    """
    Extract a 2D slice for the chosen axis and parameter index.
    """
    if ax_plot == 0:
        res_slice = res[slice_idx, :, :, param_idx].copy()
        mask_slice = mask_map[slice_idx, :, :, param_idx]
        deg_slice = mask_degeneracy[slice_idx, :, :, param_idx].copy()
    elif ax_plot == 1:
        res_slice = res[:, slice_idx, :, param_idx].copy()
        mask_slice = mask_map[:, slice_idx, :, param_idx]
        deg_slice = mask_degeneracy[:, slice_idx, :, param_idx].copy()
    elif ax_plot == 2:
        res_slice = res[:, :, slice_idx, param_idx].copy()
        mask_slice = mask_map[:, :, slice_idx, param_idx]
        deg_slice = mask_degeneracy[:, :, slice_idx, param_idx].copy()
    else:
        raise ValueError("ax_plot must be 0, 1, or 2")

    res_slice[mask_slice == 0] = np.nan
    deg_slice[mask_slice == 0] = 0

    return res_slice, mask_slice, deg_slice



# ============================================================
# Main function
# ============================================================

def main_model_fit(main_folder, results_folder, dwi_path, mask_path,
                   model="Nexi", noise="rician", hidden_layers=[50, 30], nb_simu=700_000, nb_theta=1_000):
   
    name_extension = ""
    ax_plot = 2
    n_jobs = 7
    
    if model.lower() == "nexi":
        nf_features = 14
    elif model.lower() == "sandix":
        nf_features = 22
    else:
        raise ValueError(f"Unknown model: {model}")
    
    train_size = int(nb_simu)
    
    config_file = (
        main_folder
        / "uGUIDE_config"
        / "results"
        / model
        / noise
        / f"train_size_{train_size}"
        / f"MLP_{nf_features}_features_layers_{hidden_layers[0]}_{hidden_layers[1]}"
        / f"config_{model}.pkl"
    )

    config = load_config_uGUIDE(savefile=config_file)

    # ============================================================
    # Paths
    # ============================================================
    # subject_folder = main_folder / f"{subject_id}" /  f"{session_id}" 
    # dwi_folder = subject_folder / "dwi"
    # uguide_folder = dwi_folder / f"{model}_uGUIDE"
    #results_folder = uguide_folder 
    plot_folder = results_folder / "plots"

    results_folder.mkdir(parents=True, exist_ok=True)
    plot_folder.mkdir(parents=True, exist_ok=True)

    # dwi_path = uguide_folder / "inputs" / "powderaverage_dwi.nii.gz"
    # mask_path = uguide_folder / "inputs" / "mask_dil.nii.gz"

    map_file = results_folder / f"uGUIDE_MAP_{model}{name_extension}.nii.gz"
    mask_file = results_folder / f"uGUIDE_mask_{model}{name_extension}.nii.gz"
    degeneracy_file = results_folder / f"uGUIDE_mask_degeneracy_{model}{name_extension}.nii.gz"
    uncertainty_file = results_folder / f"uGUIDE_uncertainty_{model}{name_extension}.nii.gz"
    ambiguity_file = results_folder / f"uGUIDE_ambiguity_{model}{name_extension}.nii.gz"

    print("=" * 80)
    #print(f"Subject {subject_id}, session {session_id}")
    print(f"DWI path       : {dwi_path}")
    print(f"Mask path      : {mask_path}")
    print(f"Results folder : {results_folder}")
    print("=" * 80)

    if not dwi_path.exists():
        raise FileNotFoundError(f"DWI file not found: {dwi_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    dwi_img = nib.load(dwi_path)
    dwi_data = dwi_img.get_fdata()

    mask_img = nib.load(mask_path)
    mask_data = mask_img.get_fdata()

    middle = dwi_data.shape[ax_plot] // 2

    all_map_files_exist = all(
        [
            map_file.exists(),
            mask_file.exists(),
            degeneracy_file.exists(),
            uncertainty_file.exists(),
            ambiguity_file.exists(),
        ]
    )

    # ============================================================
    # Either load existing maps or compute them
    # ============================================================
    if all_map_files_exist:
        print("All output map files already exist. Loading existing results.")

        param_map_nii = nib.load(map_file).get_fdata()
        mask_map_nii = nib.load(mask_file).get_fdata()
        degeneracy_map_nii = nib.load(degeneracy_file).get_fdata()
        uncertainty_map_nii = nib.load(uncertainty_file).get_fdata()
        ambiguity_map_nii = nib.load(ambiguity_file).get_fdata()

    else:
        print("Some output map files are missing. Computing estimates.")

        idx_mask = np.nonzero(mask_data)
        print("Number of voxels in mask:", idx_mask[0].shape[0])

        valid = np.isfinite(dwi_data[idx_mask[0], idx_mask[1], idx_mask[2], :])
        idx_mask_valid = np.where(np.sum(valid, axis=1) == dwi_data.shape[-1])[0]

        idx_valid = [
            idx_mask[0][idx_mask_valid],
            idx_mask[1][idx_mask_valid],
            idx_mask[2][idx_mask_valid],
        ]

        nb_voxels = idx_valid[0].shape[0]
        print("Number of valid voxels to estimate:", nb_voxels)
        print("Estimation of the microstructure parameters from the test signals")

        start_time = time.time()

        data_to_fit = dwi_data[
            idx_valid[0],
            idx_valid[1],
            idx_valid[2],
            :,
        ]
        
        import torch
        signal = torch.as_tensor(
            data_to_fit,
            dtype=torch.float32,
        )
        config["voxel_batch_size"] = 16
        config["nb_samples"] = 10000
        param_est, mask_est, mask_degeneracy_est, uncertainty_est, ambiguity_est = estimate_microstructure(
               signal,
               config,
               postprocessing=postprocess_NEXI_SANDIX,
               plot=False)
   

        stop_time = time.time()
        print("Time to estimate parameters in all voxels in min:", (stop_time - start_time)/60)

        n_params = len(config["prior_postprocessing"])

        # Convert flat results from torch to NumPy
        param_est = param_est.detach().cpu().numpy()
        mask_est = mask_est.detach().cpu().numpy()
        mask_degeneracy_est = mask_degeneracy_est.detach().cpu().numpy()
        uncertainty_est = uncertainty_est.detach().cpu().numpy()
        ambiguity_est = ambiguity_est.detach().cpu().numpy()
        
        nb_voxels = len(idx_valid[0])
        n_params = len(config["prior_postprocessing"])
        
        assert signal.shape[0] == nb_voxels
        assert param_est.shape == (nb_voxels, n_params)
        
        # Create full-volume maps
        param_map_nii = np.zeros((*mask_data.shape, n_params),dtype=np.float32)
        mask_map_nii = np.zeros( (*mask_data.shape, n_params), dtype=np.uint8)
        degeneracy_map_nii = np.zeros((*mask_data.shape, n_params),dtype=np.uint8)
        uncertainty_map_nii = np.zeros((*mask_data.shape, n_params),dtype=np.float32)
        ambiguity_map_nii = np.zeros((*mask_data.shape, n_params), dtype=np.float32)
        
        # Insert all valid voxels at once
        param_map_nii[idx_valid[0], idx_valid[1], idx_valid[2], :] = param_est
        mask_map_nii[idx_valid[0], idx_valid[1], idx_valid[2], :] = mask_est
        degeneracy_map_nii[idx_valid[0], idx_valid[1], idx_valid[2], :] = mask_degeneracy_est
        uncertainty_map_nii[idx_valid[0], idx_valid[1], idx_valid[2], :] = uncertainty_est
        ambiguity_map_nii[idx_valid[0], idx_valid[1], idx_valid[2], :] = ambiguity_est
        

        # param_map = np.zeros((*mask_data.shape, n_params), dtype=np.float32)
        # mask_map = np.zeros((*mask_data.shape, n_params), dtype=np.uint8)
        # mask_degeneracy = np.zeros((*mask_data.shape, n_params), dtype=np.uint8)
        # uncertainty = np.zeros((*mask_data.shape, n_params), dtype=np.float32)
        # ambiguity = np.zeros((*mask_data.shape, n_params), dtype=np.float32)

        # for idx in range(nb_voxels):
        #     i = idx_valid[0][idx]
        #     j = idx_valid[1][idx]
        #     k = idx_valid[2][idx]

        #     param_map[i, j, k, :] = estimates[idx][0]
        #     mask_map[i, j, k, :] = estimates[idx][1]
        #     mask_degeneracy[i, j, k, :] = estimates[idx][2]
        #     uncertainty[i, j, k, :] = estimates[idx][3]
        #     ambiguity[i, j, k, :] = estimates[idx][4]

        nib.Nifti1Image(param_map_nii.astype(np.float32), affine=dwi_img.affine).to_filename(map_file)
        nib.Nifti1Image(mask_map_nii.astype(np.uint8), affine=dwi_img.affine).to_filename(mask_file)
        nib.Nifti1Image(degeneracy_map_nii.astype(np.uint8), affine=dwi_img.affine).to_filename(degeneracy_file)
        nib.Nifti1Image(uncertainty_map_nii.astype(np.float32), affine=dwi_img.affine).to_filename(uncertainty_file)
        nib.Nifti1Image(ambiguity_map_nii.astype(np.float32), affine=dwi_img.affine).to_filename(ambiguity_file)

        print("Saved computed maps:")
        print(map_file)
        print(mask_file)
        print(degeneracy_file)
        print(uncertainty_file)
        print(ambiguity_file)

    # ============================================================
    # Plot results
    # ============================================================
    nb_voxels_in_mask_map = np.count_nonzero(mask_data)

    for res, res_name in zip(
        [param_map_nii, uncertainty_map_nii, ambiguity_map_nii],
        ["map", "uncertainty", "ambiguity"],
    ):
        for p, param in enumerate(config["prior_postprocessing"].keys()):

            masked_res, _, masked_deg = get_slice_and_masks(
                res=res,
                mask_map=mask_map_nii,
                mask_degeneracy=degeneracy_map_nii,
                slice_idx=middle,
                ax_plot=ax_plot,
                param_idx=p,
            )

            valid_values = masked_res[~np.isnan(masked_res)]
            if valid_values.size == 0:
                print(f"{res_name} {param}: no valid values, skipping plot")
                continue

            print(
                f"{res_name} {param}: "
                f"min = {valid_values.min()} - max = {valid_values.max()}"
            )

            if res_name == "map":
                zmin = config["prior_postprocessing"][param][0]
                zmax = config["prior_postprocessing"][param][1]
            else:
                zmin = 0
                zmax = 50

            fig = px.imshow(
                masked_res.T,
                zmin=zmin,
                zmax=zmax,
                origin="lower",
                color_continuous_scale="viridis",
            )

            idx_deg = np.nonzero(masked_deg.T)
            fig.add_trace(
                go.Scatter(
                    x=idx_deg[1],
                    y=idx_deg[0],
                    mode="markers",
                    marker_color="red",
                    showlegend=False,
                )
            )

            fig.update_layout(
                plot_bgcolor="rgba(255, 255, 255, 0)",
                paper_bgcolor="rgba(255, 255, 255, 1)",
                font={"size": 24},
                coloraxis_colorbar_x=0.87,
            )
            fig.update_xaxes(showticklabels=False)
            fig.update_yaxes(showticklabels=False)

            plot_file = plot_folder / f"plot_{res_name}_{param}_masked_{model}_ax{ax_plot}.png"
            fig.write_image(plot_file)
            print(f"Saved plot: {plot_file}")

    # ============================================================
    # Degeneracy summary
    # ============================================================
    for p, param in enumerate(config["prior_postprocessing"].keys()):
        deg_mask_param = degeneracy_map_nii[..., p].copy()
        deg_mask_param[mask_map_nii[..., p] == 0] = 0
        nb_deg = np.count_nonzero(deg_mask_param)
        print(f"{param}: {nb_deg}/{nb_voxels_in_mask_map} degeneracies")
