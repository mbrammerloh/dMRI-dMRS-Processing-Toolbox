import numpy as np
from pathlib import Path
from tqdm import tqdm
import nibabel as nib
from joblib import Parallel, delayed
import graymatter_swissknife as gmsk
from graymatter_swissknife.models.parameters.acq_parameters import AcquisitionParameters
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# ============================================================
# Options
# ============================================================
add_noise = True
plot_distribs = True
save_both_rician_and_noiseless = True
# Must have add_noise = True to save both rician and noiseless


# ============================================================
# Helper functions
# ============================================================
def add_rice_noise(signal, sigma_array, nb_directions):
    """
    Add Rician noise.
    signal shape: (n_voxels, n_measurements)
    sigma_array shape: (n_voxels,)
    nb_directions: int or array-like of length n_measurements
    """
    if np.isscalar(nb_directions):
        if not isinstance(nb_directions, int):
            raise TypeError("nb_directions should be an integer when scalar.")

        if nb_directions > 1:
            sigma = np.tile(sigma_array.reshape(signal.shape[0], 1),
                            (1, signal.shape[1]))
            signal_snr_in_multiple_directions = np.zeros(signal.shape + (nb_directions,))
            for i in range(nb_directions):
                dist1 = np.random.randn(signal.shape[0], signal.shape[1])
                dist2 = np.random.randn(signal.shape[0], signal.shape[1])
                signal_snr = np.sqrt((signal + dist1 * sigma) ** 2 +
                                     (dist2 * sigma) ** 2)
                signal_snr_in_multiple_directions[:, :, i] = signal_snr
            signal_snr = np.mean(signal_snr_in_multiple_directions, axis=-1)

        elif nb_directions == 1:
            sigma = np.tile(sigma_array.reshape(signal.shape[0], 1),
                            (1, signal.shape[1]))
            dist1 = np.random.randn(signal.shape[0], signal.shape[1])
            dist2 = np.random.randn(signal.shape[0], signal.shape[1])
            signal_snr = np.sqrt((signal + dist1 * sigma) ** 2 +
                                 (dist2 * sigma) ** 2)
        else:
            raise ValueError("Number of directions should be strictly positive")

    else:
        nb_directions = np.asarray(nb_directions)
        if len(nb_directions) != signal.shape[1]:
            raise ValueError(
                "nb_directions should have length equal to the number of measurements"
            )

        signal_snr = np.zeros(signal.shape)
        for k, nb_dir_k in enumerate(nb_directions):
            signal_snr_bval_k = np.zeros((signal.shape[0], nb_dir_k))
            for i in range(nb_dir_k):
                dist1 = np.random.randn(signal.shape[0])
                dist2 = np.random.randn(signal.shape[0])
                signal_snr_bval_k_dir_i = np.sqrt(
                    (signal[:, k] + dist1 * sigma_array) ** 2 +
                    (dist2 * sigma_array) ** 2
                )
                signal_snr_bval_k[:, i] = signal_snr_bval_k_dir_i
            signal_snr[:, k] = np.mean(signal_snr_bval_k, axis=-1)

    return signal_snr


def plot_parameter_histograms(parameters, title_prefix="Parameter"):
    """
    Plot histograms of model parameters.
    """
    n_param = parameters.shape[1]
    ncols = min(4, n_param)
    nrows = int(np.ceil(n_param / ncols))

    fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axs = np.atleast_1d(axs).flatten()

    for i in range(n_param):
        axs[i].hist(parameters[:, i], bins=100)
        axs[i].set_title(f"{title_prefix} {i}")

    for j in range(n_param, len(axs)):
        axs[j].axis("off")

    plt.tight_layout()
    plt.show()
    plt.close(fig)


def plot_sigma_histogram(sigma_all):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.hist(sigma_all, bins=100, range=(0, 0.1))
    ax.set_title("Noise")
    plt.show()
    plt.close(fig)


def eval_mist_model(param_irunning, mist_model_irunning, acq_param_irunning):
    return mist_model_irunning.get_signal(param_irunning, acq_param_irunning)


# ============================================================
# Build / load sigma distribution
# ============================================================

def simulate_data(main_folder, b, delta, nb_directions, small_delta, sigma_files, mask_files, model_list, adjust_parameter_limits=None):
    uGUIDE_folder = main_folder / "uGUIDE_config"
    uGUIDE_folder.mkdir(parents=True, exist_ok=True)
    
    sigma_filename = uGUIDE_folder / "sigma.npz"
    save_in_folder = uGUIDE_folder / "simulations"
    save_in_folder.mkdir(parents=True, exist_ok=True)
    
 
    if add_noise:
        noise_model = "rician"
        sigma_list = []
    
        for (sigma_file,mask_file) in zip(sigma_files,mask_files):
            
            sigma_file = Path(sigma_file)
            mask_file = Path(mask_file)

            if not sigma_file.exists():
                raise FileNotFoundError(f"Sigma file not found: {sigma_file}")
            if not mask_file.exists():
                raise FileNotFoundError(f"Mask file not found: {mask_file}")
    
            cortical_mask = nib.load(mask_file).get_fdata() > 0
            sigma = nib.load(sigma_file).get_fdata()
            sigma_list.append(sigma[cortical_mask])
    
        sigma_all = np.concatenate(sigma_list)
        
        # Remove Nans
        sigma_all = sigma_all[~np.isnan(sigma_all)]
    
        # Drop first and last 0.1 percentiles
        sigma_all = sigma_all[
            (sigma_all > np.percentile(sigma_all, 0.1))
            & (sigma_all < np.percentile(sigma_all, 99.9))
        ]
    
        np.savez(sigma_filename, sigma_all=sigma_all)
    
        # Also reload from disk if you want a single source of truth
        sigma_all = np.load(sigma_filename)["sigma_all"]
    
    else:
        noise_model = "noiseless"
        sigma_all = None
    
    # ============================================================
    # Main simulation loop
    # ============================================================
    
    
    acq_parameters = AcquisitionParameters(b, delta, small_delta)
    
    # --------------------------------------------------------
    # Loop over microstructure models
    # --------------------------------------------------------
    for mist_model_name in model_list:
        print(f"Running for model: {mist_model_name}")
    
        upper_mist_name = mist_model_name.upper()
        model_module = getattr(gmsk.models, upper_mist_name)
        model_submodule = getattr(model_module, mist_model_name.lower())
        mist_model_class = getattr(model_submodule, mist_model_name)
        mist_model = mist_model_class()
    
        param_name = mist_model.param_names
        n_param = mist_model.n_params
        limits = mist_model.classic_limits
        
        if 'adjust_parameter_limits' in locals() and adjust_parameter_limits is not None:
            limits= adjust_parameter_limits
    
        print(f"Selected model: {mist_model_name}")
        print(f"Parameter names: {param_name}")
        print(f"Parameter limits:\n{limits}")
    
        # ----------------------------------------------------
        # Loop over train / valid / test sets
        # ----------------------------------------------------
        for train_or_test in ["train", "valid", "test"]:
            print(f"Running for {train_or_test} set")
    
            filename = (
                save_in_folder
                / f"model-{mist_model_name}_noise-c1_limits-classic_noise-{noise_model}_set-{train_or_test}.npz"
            )
            print(f"Data will be saved in {filename}")
    
            if train_or_test == "train":
                random_seed = 42
                n_voxels = int(1e6)
            elif train_or_test == "valid":
                random_seed = 67
                n_voxels = int(1e4)
            elif train_or_test == "test":
                random_seed = 68
                n_voxels = int(1e4)
            else:
                raise ValueError("train_or_test should be either 'train', 'valid' or 'test'")
    
            # ------------------------------------------------
            # Random parameters
            # ------------------------------------------------
            np.random.seed(random_seed)
    
            parameters = np.random.rand(n_voxels, n_param) * (
                limits[:, 1] - limits[:, 0]
            ) + limits[:, 0]
    
            #Dmin, Dmax = 0.1, 3.5
            Di_idx = param_name.index('Di')
            Dmin, Dmax = limits[Di_idx]

            Di_samples = np.zeros(n_voxels)
            De_samples = np.zeros(n_voxels)
            U0_samples = np.zeros(n_voxels)
            U1_samples = np.zeros(n_voxels)
    
            for i in range(n_voxels):
                u0 = np.random.rand()
                u1 = np.random.rand()
    
                Di = Dmin + np.sqrt((Dmax - Dmin) ** 2 * u0)
                De = Dmin + (Di - Dmin) * u1
    
                Di_samples[i] = Di
                De_samples[i] = De
                U0_samples[i] = u0
                U1_samples[i] = u1
    
            D1_name = "Di" if "Di" in param_name else "Dd"
            D2_name = "De"
    
            D1_index = param_name.index(D1_name)
            D2_index = param_name.index(D2_name)
    
            parameters[:, D1_index] = Di_samples
            parameters[:, D2_index] = De_samples
    
            U0_U1_samples = np.stack((U0_samples, U1_samples), axis=1)
    
            if plot_distribs:
                plot_parameter_histograms(parameters)
    
            # ------------------------------------------------
            # Generate signal
            # ------------------------------------------------
            n_jobs = -1
            signal_parallel = Parallel(n_jobs=n_jobs)(
                delayed(eval_mist_model)(
                    parameters[irunning, :],
                    mist_model,
                    acq_parameters,
                )
                for irunning in tqdm(range(n_voxels))
            )
            signal_parallel = np.array(signal_parallel)
    
            # ------------------------------------------------
            # Add noise
            # ------------------------------------------------
            if add_noise:
                if sigma_all is None:
                    raise ValueError("sigma_all is missing although add_noise=True")
    
                if plot_distribs:
                    plot_sigma_histogram(sigma_all)
    
                np.random.seed(random_seed + 9)
                sigma_chosen = np.random.choice(sigma_all, n_voxels)
    
                signal_parallel_noisemodel = add_rice_noise(
                    signal_parallel,
                    sigma_chosen,
                    nb_directions=nb_directions,
                )
            else:
                signal_parallel_noisemodel = signal_parallel
                sigma_chosen = None
    
            # ------------------------------------------------
            # Save noisy data
            # ------------------------------------------------
            np.savez(
                filename,
                signal=signal_parallel_noisemodel,
                parameters=parameters,
                U0_U1_samples=U0_U1_samples,
                noise_model=noise_model,
                sigma=sigma_chosen,
                parameter_names=param_name,
                limits=limits,
                U0_U1_names=["U0", "U1"],
                explanation=(
                    "U0 and U1 are the random uniform variables used to generate "
                    "Di and De using the formula: "
                    "Di = Dmin + sqrt((Dmax - Dmin)**2 * U0) and "
                    "De = Dmin + (Di - Dmin) * U1"
                ),
            )
    
            # ------------------------------------------------
            # Save noiseless version too
            # ------------------------------------------------
            if add_noise and save_both_rician_and_noiseless:
                filename_noiseless = (
                    save_in_folder
                    / f"model-{mist_model_name}_noise-c1_limits-classic_noise-noiseless_set-{train_or_test}.npz"
                )
                print(f"Saving associated noiseless data in {filename_noiseless}")
    
                np.savez(
                    filename_noiseless,
                    signal=signal_parallel,
                    parameters=parameters,
                    U0_U1_samples=U0_U1_samples,
                    noise_model="noiseless",
                    sigma=None,
                    parameter_names=param_name,
                    limits=limits,
                    U0_U1_names=["U0", "U1"],
                    explanation=(
                        "U0 and U1 are the random uniform variables used to generate "
                        "Di and De using the formula: "
                        "Di = Dmin + sqrt((Dmax - Dmin)**2 * U0) and "
                        "De = Dmin + (Di - Dmin) * U1"
                    ),
                )