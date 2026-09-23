import time
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from uGUIDE.config_utils import create_config_uGUIDE, save_config_uGUIDE
from uGUIDE.data_utils import preprocess_data
#from uGUIDE.inference import run_inference
from uGUIDE.training import run_training
from uGUIDE.estimation import estimate_microstructure



# ============================================================
# Helper functions
# ============================================================
def postprocess_NEXI_SANDIX(samples, config):
    """
    Convert U0/U1 into Di/De.
    For SANDIX, also convert f and fs into neurite/soma fractions.
    """
    prior_keys = np.array(list(config["prior"].keys()))

    idx_u0 = np.where(prior_keys == "U0")[0]
    idx_u1 = np.where(prior_keys == "U1")[0]

    u0 = samples[:, idx_u0]
    u1 = samples[:, idx_u1]

    u0 = np.clip(u0, 0, 1)
    u1 = np.clip(u1, 0, 1)

    Di_min = 0.1
    Di_max = 3.5
    De_min = 0.1

    Di = np.sqrt((Di_max - Di_min) ** 2 * u0) + Di_min
    De = (Di - Di_min) * u1 + De_min

    out_samples = samples.copy()
    out_samples[:, idx_u0] = Di
    out_samples[:, idx_u1] = De

    if config["microstructure_model_name"].lower() == "sandix":
        idx_f = np.where(prior_keys == "f")[0]
        idx_fs = np.where(prior_keys == "fs")[0]

        f = samples[:, idx_f]
        fs = samples[:, idx_fs]

        f_neurites = f * (1 - fs)
        f_soma = f * fs

        out_samples[:, idx_f] = f_neurites
        out_samples[:, idx_fs] = f_soma

    return out_samples


def plot_prior_distribution(samples, prior, savename="prior_distribution.png", savedir=None):
    """
    Plot and optionally save histograms of parameter distributions.
    """
    if samples.shape[1] != len(prior):
        raise ValueError("Number of sample columns and prior parameters do not match.")

    fig, axs = plt.subplots(1, len(prior), figsize=(5 * len(prior), 5))

    if len(prior) == 1:
        axs = [axs]

    for i, param in enumerate(prior.keys()):
        axs[i].hist(samples[:, i], bins=50)
        axs[i].set_title(param)
        axs[i].set_xlim(prior[param][0], prior[param][1])

    if savedir is not None:
        savedir.mkdir(parents=True, exist_ok=True)
        fig.savefig(savedir / savename, bbox_inches="tight")

    plt.show()
    plt.close(fig)


def filter_di_de_below_3(x, theta_uniform, theta_postproc):
    """
    Keep only samples with Di <= 3 and De <= 3.
    Assumes Di and De are columns 1 and 2 in theta_postproc.
    """
    keep = (theta_postproc[:, 1] <= 3) & (theta_postproc[:, 2] <= 3)
    return x[keep, :], theta_uniform[keep, :], theta_postproc[keep, :]


# ============================================================
# Main function
# ============================================================
def model_training(main_folder, model="Nexi", noise="rician", hidden_layers=[50, 30], nb_simu=700_000, nb_theta=1_000, adjust_parameter_limits=None):
        
    if model.lower() == "nexi":
        nf_features = 14
    elif model.lower() == "sandix":
        nf_features = 22
    else:
        raise ValueError(f"Unknown model: {model}")
        
    uGUIDE_folder = main_folder / "uGUIDE_config"
    simulation_folder = uGUIDE_folder / "simulations"
    
    train_sim_data_path = (
        simulation_folder
        / f"model-{model}_noise-c1_limits-classic_noise-{noise}_set-train.npz"
    )
    test_sim_data_path = (
        simulation_folder
        / f"model-{model}_noise-c1_limits-classic_noise-{noise}_set-test.npz"
    )
    
    if not train_sim_data_path.exists():
        raise FileNotFoundError(f"Training simulation file not found: {train_sim_data_path}")
    if not test_sim_data_path.exists():
        raise FileNotFoundError(f"Test simulation file not found: {test_sim_data_path}")


    # ============================================================
    # Load training data
    # ============================================================
    npz_train = np.load(train_sim_data_path, allow_pickle=True)
    
    x_train = npz_train["signal"]
    
    theta_postproc_train = npz_train["parameters"]
    theta_uniform_train = theta_postproc_train.copy()
    theta_uniform_train[:, 1:3] = npz_train["U0_U1_samples"]
    
    #x_train, theta_uniform_train, theta_postproc_train = filter_di_de_below_3(
      #  x_train, theta_uniform_train, theta_postproc_train
    #) #rita
    
    print(f"Theta train shape: {theta_postproc_train.shape}")
    print(f"x train shape: {x_train.shape}")


    # ============================================================
    # Preprocess training data
    # ============================================================
    theta_train, x_train = preprocess_data(
        theta_uniform_train[:nb_simu, :],
        x_train[:nb_simu, :],
        bvals=[],
        normalize=False,
    )
    
    print(f"Theta train shape after preprocessing: {theta_train.shape}")
    print(f"x train shape after preprocessing: {x_train.shape}")
    
    train_size = theta_train.shape[0]
    
    
    # ============================================================
    # Create results folder
    # ============================================================
    base_results_folder = (
        uGUIDE_folder
        / "results"
        / model
        / noise
        / f"train_size_{train_size}"
    )
    
    if nf_features == 0:
        use_MLP = False
        folder_save = base_results_folder / "no_MLP"
    else:
        use_MLP = True
        folder_save = (
            base_results_folder
            / f"MLP_{nf_features}_features_layers_{hidden_layers[0]}_{hidden_layers[1]}"
        )
    
    folder_save.mkdir(parents=True, exist_ok=True)
    
    print(f"Results folder: {folder_save}")
    
    
    # ============================================================
    # Load priors from training data
    # ============================================================
    prior_postproc_params = npz_train["parameter_names"]
    prior_uniform_params = prior_postproc_params.copy()
    prior_uniform_params[1:3] = npz_train["U0_U1_names"]
    
    prior_postproc_lim = npz_train["limits"].copy()
    # prior_postproc_lim[1][1] = 3.0 # rita
    # prior_postproc_lim[2][1] = 3.0 # rita
    
    prior_uniform_lim = prior_postproc_lim.copy()
    # prior_uniform_lim[1] = [0.0, 9 / (3.5) ** 2]
    prior_uniform_lim[1] = [0.0, 1.0]  # rita: allows Di up to 3.5
    prior_uniform_lim[2] = [0.0, 1.0]
    
    prior_uniform = {
        p: [prior_uniform_lim[i, 0], prior_uniform_lim[i, 1]]
        for i, p in enumerate(prior_uniform_params)
    }
    print(f"Uniform prior: {prior_uniform}")
    
    plot_prior_distribution(
        theta_uniform_train,
        prior_uniform,
        savename="prior_uniform_distribution.png",
        savedir=folder_save,
    )
    
    prior_postprocessing = {
        p: [prior_postproc_lim[i, 0], prior_postproc_lim[i, 1]]
        for i, p in enumerate(prior_postproc_params)
    }
    print(f"Prior postprocessing: {prior_postprocessing}")
    
    plot_prior_distribution(
        theta_postproc_train,
        prior_postprocessing,
        savename="prior_postprocessing_distribution.png",
        savedir=folder_save,
    )
    
    
    # ============================================================
    # Create and save config
    # ============================================================
    config = create_config_uGUIDE(
        microstructure_model_name=model,
        folderpath=folder_save,
        size_x=x_train.shape[1],
        prior=prior_uniform,
        prior_postprocessing=prior_postprocessing,
        device="cuda",
        use_MLP=use_MLP,
        nf_features=nf_features,
        hidden_layers=hidden_layers,
        max_epochs=1000,
        n_epochs_no_change=25,
        random_seed=4237,
        nb_samples=50_000,
    )
    
    print(f'Device: {config["device"]}')
    
    config_path = folder_save / f"config_{model}.pkl"
    save_config_uGUIDE(config, savefile=config_path)
    print(f"Saved config to: {config_path}")
    
    
    # ============================================================
    # Train uGUIDE
    # ============================================================
    
    print("Training uGUIDE...")
    config["learning_rate"] = 1e-4
    run_training(
        theta_train,
        x_train,
        config=config,
        plot_loss=True,
        load_state=False,
    )
    
    # config["learning_rate"] = 1e-4
    # run_inference(
    #     theta_train,
    #     x_train,
    #     config=config,
    #     plot_loss=False,
    #     load_state=True,
    # )
    
    
    # ============================================================
    # Load test data
    # ============================================================
    npz_test = np.load(test_sim_data_path, allow_pickle=True)
    
    x_test = npz_test["signal"]
    
    theta_postproc_test = npz_test["parameters"]
    theta_uniform_test = theta_postproc_test.copy()
    theta_uniform_test[:, 1:3] = npz_test["U0_U1_samples"]
    
    # x_test, theta_uniform_test, theta_postproc_test = filter_di_de_below_3(
     #   x_test, theta_uniform_test, theta_postproc_test
    #) # rita
    
    print(f"Theta test shape: {theta_postproc_test.shape}")
    print(f"x test shape: {x_test.shape}")
    
    
    # ============================================================
    # Preprocess test data
    # ============================================================
    theta_test, x_test = preprocess_data(
        theta_uniform_test,
        x_test,
        bvals=[],
        normalize=False,
    )
    
    print(f"Theta test shape after preprocessing: {theta_test.shape}")
    print(f"x test shape after preprocessing: {x_test.shape}")
    
    
    # ============================================================
    # Check train/test prior consistency
    # ============================================================
    prior_postproc_params_test = npz_test["parameter_names"]
    prior_uniform_params_test = prior_postproc_params_test.copy()
    prior_uniform_params_test[1:3] = npz_test["U0_U1_names"]
    
    prior_postproc_lim_test = npz_test["limits"].copy()
    #prior_postproc_lim_test[1][1] = 3.0 # rita
    #prior_postproc_lim_test[2][1] = 3.0 # rita
    
    prior_uniform_lim_test = prior_postproc_lim_test.copy()
    #prior_uniform_lim_test[1] = [0.0, 9 / (3.5) ** 2] #rita
    prior_uniform_lim_test[1] = [0.0, 1.0]
    prior_uniform_lim_test[2] = [0.0, 1.0]
    
    assert (prior_uniform_params == prior_uniform_params_test).all()
    assert np.allclose(prior_uniform_lim, prior_uniform_lim_test)
    assert (prior_postproc_params == prior_postproc_params_test).all()
    assert np.allclose(prior_postproc_lim, prior_postproc_lim_test)
    
    
    # ============================================================
    # Quick estimation checks on a few test voxels
    # ============================================================
    print("Test estimation on a few voxels:")
    n_check = min(5, len(x_test))
    
    # for i in range(n_check):
    #     _ = estimate_microstructure(
    #         x_test[i, :],
    #         config,
    #         plot=True,
    #         #voxel_id=i,
    #         postprocessing=None,
    #         theta_gt=theta_uniform_test[i, :],
    #     )
    #     _ = estimate_microstructure(
    #         x_test[i, :],
    #         config,
    #         plot=True,
    #         #voxel_id=i,
    #         postprocessing=postprocess_NEXI_SANDIX,
    #         theta_gt=theta_postproc_test[i, :],
    #     )
    
    _ = estimate_microstructure(x_test[:n_check, :],
                            config,
                            verbose=True,
                            plot=True,
                            postprocessing=None,
                            theta_gt=theta_test[:n_check, :])
    _ = estimate_microstructure(x_test[:n_check, :],
                            config,
                            verbose=True,
                            plot=True,
                            postprocessing=postprocess_NEXI_SANDIX,
                            theta_gt=theta_postproc_test[:n_check :])


    
    # ============================================================
    # Plot test prior distributions
    # ============================================================
    nb_theta = min(nb_theta, len(x_test))
    
    plot_prior_distribution(
        theta_uniform_test[:nb_theta, :],
        prior_uniform,
        savename="prior_uniform_distribution_test.png",
        savedir=folder_save,
    )
    plot_prior_distribution(
        theta_postproc_test[:nb_theta, :],
        prior_postprocessing,
        savename="prior_postprocessing_distribution_test.png",
        savedir=folder_save,
    )
    
    
    # ============================================================
    # Estimate parameters on the test set
    # ============================================================
    print("Estimation of the microstructure parameters from the test signals")
    start_time = time.time()
    
    # estimates = Parallel(n_jobs=-1)(
    #     delayed(estimate_microstructure)(
    #         x_test[i, :],
    #         config,
    #         postprocessing=postprocess_NEXI_SANDIX,
    #         voxel_id=i,
    #         theta_gt=theta_postproc_test[i, :],
    #         plot=False,
    #     )
    #     for i in range(nb_theta)
    # )
    param_map, mask, mask_degeneracy, uncertainty, ambiguity = estimate_microstructure(
            x_test[:nb_theta, :],
            config,
            postprocessing=postprocess_NEXI_SANDIX,
            theta_gt=theta_postproc_test[:nb_theta, :],
            plot=False,
        )
    
    
    stop_time = time.time()
    print("Time to estimate parameters in all voxels:", stop_time - start_time)
    
    
    # ============================================================
    # Extract estimates
    # ============================================================
    # param_map = np.zeros((nb_theta, config["size_theta"]))
    # mask = np.zeros((nb_theta, config["size_theta"]), dtype=bool)
    # mask_degeneracy = np.zeros((nb_theta, config["size_theta"]), dtype=bool)
    # uncertainty = np.zeros((nb_theta, config["size_theta"]))
    # ambiguity = np.zeros((nb_theta, config["size_theta"]))
    
    # for i in range(nb_theta):
    #     param_map[i, :] = estimates[0][i]
    #     mask[i, :] = estimates[1][i]
    #     mask_degeneracy[i, :] = estimates[2][i]
    #     uncertainty[i, :] = estimates[3][i]
    #     ambiguity[i, :] = estimates[4][i]
    
    
    # ============================================================
    # Print number of degeneracies per parameter
    # ============================================================
    print("Number of degeneracies per parameter:")
    for p, param in enumerate(config["prior_postprocessing"].keys()):
        nb_deg = np.count_nonzero(mask_degeneracy[:, p])
        print(f"{param}: {nb_deg}/{mask_degeneracy.shape[0]} degeneracies")
        print(f"{param}: {100 * nb_deg / mask_degeneracy.shape[0]:.2f}% degeneracies")
    
    
    # ============================================================
    # Figures 3 & 4
    # ============================================================
    fontsize = 30
    ptsize = 20
    
    fig, ax = plt.subplots(
        2,
        config["size_theta"],
        gridspec_kw={"height_ratios": [2, 1]},
        figsize=(20, 12),
        layout="constrained",
    )
    
    cmap = plt.get_cmap("turbo")
    vmin = 0
    vmax = 50
    cmap_norm = plt.Normalize(vmin=0, vmax=vmax)
    colors = cmap(np.linspace(0, 0.75, cmap.N))
    cmap_cut = matplotlib.colors.LinearSegmentedColormap.from_list("turbo_cut", colors)
    
    for idx, p in enumerate(config["prior_postprocessing"].keys()):
        vox_mask = np.where(mask[:, idx])[0]
        vox_non_deg = np.where(~mask_degeneracy[:, idx])[0]
        vox = np.intersect1d(vox_mask, vox_non_deg)
    
        zero_estimates = np.where(param_map[:, idx] == 0)[0]
        vox = np.setdiff1d(vox, zero_estimates)
    
        ax[0, idx].plot(
            theta_postproc_test[vox, idx],
            theta_postproc_test[vox, idx],
            alpha=0.5,
        )
        ax[0, idx].scatter(
            theta_postproc_test[vox, idx],
            param_map[vox, idx],
            s=ptsize,
            color=cmap_cut(cmap_norm(uncertainty[vox, idx])),
        )
    
        vox_deg = np.where(mask_degeneracy[:, idx])[0]
        vox_deg = np.intersect1d(vox_mask, vox_deg)
    
        ax[0, idx].scatter(
            theta_postproc_test[vox_deg, idx],
            param_map[vox_deg, idx],
            s=ptsize,
            c="r",
        )
    
        if p == "t_ex":
            ax[0, idx].set_title(r"$t_{ex}$", fontsize=fontsize)
        elif p == "Di":
            ax[0, idx].set_title(r"$D_{i}$", fontsize=fontsize)
        elif p == "De":
            ax[0, idx].set_title(r"$D_{e}$", fontsize=fontsize)
        else:
            ax[0, idx].set_title(fr"${p}$", fontsize=fontsize)
    
        ax[0, idx].set_xlabel("Ground truth", fontsize=fontsize)
    
        if model.lower() == "sandix" and p == "f":
            ax[0, idx].set_xticks([0.25, 0.75])
    
        ax[0, idx].tick_params(axis="both", which="major", labelsize=fontsize - 2)
    
        if idx == 0:
            ax[0, idx].set_ylabel("MAP", fontsize=fontsize)
    
        if len(vox) > 0:
            print(
                f"Mean and maximum uncertainty for {p}: "
                f"{uncertainty[vox, idx].mean():.2f}% and {uncertainty[vox, idx].max():.2f}%"
            )
            hist, bins = np.histogram(uncertainty[vox, idx], bins=30)
            x_hist = (bins[1:] + bins[:-1]) / 2
            ax[1, idx].bar(x_hist, hist, color=cmap_cut(cmap_norm(x_hist)))
        else:
            print(f"No valid non-degenerate voxels available for {p}")
            ax[1, idx].bar([], [])
    
        ax[1, idx].set_xlim(0, vmax)
        ax[1, idx].set_xlabel("%", fontsize=fontsize)
        ax[1, idx].tick_params(axis="both", which="major", labelsize=fontsize - 2)
    
        if idx == 0:
            ax[1, idx].set_ylabel("Uncertainty", fontsize=fontsize)
    
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=cmap_norm, cmap=cmap_cut),
        ax=ax[0, -1],
    )
    cbar.set_label(label="Uncertainty (%)", fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)
    
    fig.savefig(
        folder_save / f"Figure_{model}_{noise}_map_vs_gt_uncertnty_ambiguity_testset_{nb_theta}_voxels.png",
        bbox_inches="tight",
    )
    
    plt.show()
    plt.close(fig)
