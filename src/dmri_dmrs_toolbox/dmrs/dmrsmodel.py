from dmri_dmrs_toolbox.dmrs.dmrsdata import DMRSDataset
import numpy as np
import pandas as pd
from scipy.special import erf
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Dict, Any, Sequence
import copy

normalization_index=0 # 0 for normalizing by lowest b-value, 1 for second lowest, etc.

def sphere_murdaycotts(r, D0, b, big_delta, small_delta):
    """
    Calculates diffusion attenuation, mlnS = - ln(S/S0), inside a perfectly reflecting sphere of radius r, free
    diffusion coefficient D0, bvalue b (in IS units of s/m), with pulse width delta and distance big_delta between the fronts
    of the pulses, according to Murday and Cotts, JCP 1968
    Reference value: g = 0.01070 for 40 mT/m

    Derived and optimized from (c) Dmitry Novikov, June 2021
    Quentin Uhl, July 2023

    :param r: The radius of the sphere (in microns)
    :param D0: The diffusion coefficient inside the sphere (in µm²/ms)
    :param big_delta: The time of the second pulse (in ms)
    :param small_delta: The pulse width (in ms)
    :param b: The b-value (in ms/µm²)
    :return mlnS: The diffusion attenuation, mlnS = - ln(S/S0)
    """
    # Make sure all inputs are numpy arrays
    b, big_delta, small_delta = np.array(b), np.array(big_delta), np.array(small_delta)
    # Define reference values
    g = np.sqrt(b / (big_delta - small_delta / 3)) / small_delta  # in 1/µm*ms
    t_ref = r ** 2 / D0  # Compute t_ref (in ms)
    bardelta = np.expand_dims(small_delta / t_ref, axis=-1)  # Compute bardelta (unitless)
    bar_bigdelta = np.expand_dims(big_delta / t_ref, axis=-1)  # Compute bar_bigdelta (unitless)
    # Spherical roots precomputed
    alpha = np.array([2.0815759778181, 5.940369990572712, 9.205840142936665, 12.404445021901974, 15.579236410387185,
                      18.742645584774756, 21.899696479492778, 25.052825280992952, 28.203361003952356,
                      31.352091726564478, 34.499514921366952, 37.645960323086392, 40.791655231271882,
                      43.936761471419779, 47.081397412154182, 50.225651649183071, 53.369591820490818,
                      56.513270462198577, 59.656729003527936, 62.800000556519777, 65.94311190465524, 69.086084946645187,
                      72.228937762015434, 75.371685409287323, 78.514340531930912, 81.656913824036792,
                      84.799414392202507, 87.941850039659869, 91.084227491468639, 94.226552574568288,
                      97.368830362900979, 100.511065295271166, 103.653261271734152, 106.795421732944149,
                      109.937549725876437, 113.079647958579201, 116.221718846032573, 119.363764548756691,
                      122.505787005472015, 125.647787960854458, 128.789768989223461, 131.931731514842539,
                      135.07367682938397, 138.215606107009307, 141.357520417436831, 144.49942073730486,
                      147.641307960078819, 150.783182904723986, 153.925046323311989, 157.066898907714517,
                      160.208741295510009, 163.350574075206424, 166.492397790873781, 169.634212946261414,
                      172.776020008465338, 175.917819411202572, 179.059611557740794, 182.201396823524391,
                      185.343175558533943, 188.48494808940859, 191.626714721360713, 194.768475739904318,
                      197.910231412418966])
    # Put the alphas in the last dimension
    alpha_shape = alpha.shape
    for dim in range(np.max(np.ndim(big_delta), np.ndim(small_delta))):
        alpha_shape = (1,) + alpha_shape
    alpha = np.reshape(alpha, alpha_shape)
    # Compute the signal
    mlnS = np.sum((2 / (alpha ** 6 * (alpha ** 2 - 2))) * (-2 + 2 * alpha ** 2 * bardelta +
                                                           2 * (np.exp(-alpha ** 2 * bardelta) +
                                                                np.exp(-alpha ** 2 * bar_bigdelta)) -
                                                           np.exp(-alpha ** 2 * (bardelta + bar_bigdelta)) -
                                                           np.exp(-alpha ** 2 * (bar_bigdelta - bardelta))), axis=-1)
    signal = np.exp(-mlnS * D0 * g ** 2 * t_ref ** 3)
    return signal

def cylinder_perpendicular_signal(r, D0, b, big_delta, small_delta):
    """
    Calculates diffusion attenuation, mlnS = - ln(S/S0), inside a perfectly reflecting cylinder of radius r, free
    diffusion coefficient D0, bvalue b (in IS units of s/m), with pulse width delta and distance big_delta between the fronts
    of the pulses, according to Murday and Cotts, JCP 1968
    (c) Quentin Uhl, July 2023

    :param r: The radius of the cylinder (in microns)
    :param D0: The diffusion coefficient inside the cylinder (in µm²/ms)
    :param big_delta: The time of the second pulse (in ms)
    :param small_delta: The pulse width (in ms)
    :param b: The b-value (in ms/µm²)
    :return mlnS: The diffusion attenuation, mlnS = - ln(S/S0)
    """
    # Make sure all inputs are numpy arrays
    b, big_delta, small_delta = np.array(b), np.array(big_delta), np.array(small_delta)
    # Define reference values
    g = np.sqrt(b / (big_delta - small_delta / 3)) / small_delta  # in 1/µm*ms
    t_ref = r ** 2 / D0  # Compute t_ref (in ms)
    bar_delta = np.expand_dims(small_delta / t_ref, axis=-1)  # Compute bar_delta (unitless)
    bar_bigdelta = np.expand_dims(big_delta / t_ref, axis=-1)  # Compute bar_bigdelta (unitless)
    # Cylindrical roots precomputed
    alpha = np.array([1.841183781340659, 5.331442773525032, 8.536316366346286, 11.706004902592063, 14.863588633909032,
                      18.015527862681804, 21.164369859188788, 24.311326857210776, 27.457050571059245,
                      30.601922972669094, 33.746182898667385, 36.889987409236809, 40.033444053350678,
                      43.17662896544882, 46.319597561173914, 49.462391139702753, 52.605041111556687,
                      55.747571792251009, 58.890002299185703, 62.032347870661987, 65.174620802544453,
                      68.316831125951808, 71.458987105850994, 74.601095613456408, 77.743162408196767,
                      80.885192353878438, 84.027189586293531, 87.169157644540277, 90.311099574903423,
                      93.45301801376003, 96.594915254291138, 99.736793300573908, 102.878653911754455,
                      106.020498638360806, 109.162328852340863, 112.304145772055051, 115.44595048318557,
                      118.587743956319926, 121.729527061810202, 124.871300582387889])
    # Put the alphas in the last dimension
    alpha_shape = alpha.shape
    for dim in range(np.max(np.ndim(big_delta), np.ndim(small_delta))):
        alpha_shape = (1,) + alpha_shape
    alpha = np.reshape(alpha, alpha_shape)
    # Compute the signal
    mlnS = np.sum((2 / (alpha ** 6 * (alpha ** 2 - 1))) * (-2 + 2 * alpha ** 2 * bar_delta +
                                                           2 * (np.exp(-alpha ** 2 * bar_delta) +
                                                                np.exp(-alpha ** 2 * bar_bigdelta)) -
                                                           np.exp(-alpha ** 2 * (bar_delta + bar_bigdelta)) -
                                                           np.exp(-alpha ** 2 * (bar_bigdelta - bar_delta))), axis=-1)
    signal = np.exp(-mlnS * D0 * g ** 2 * t_ref ** 3)
    return signal


def sticks_signal(params, b_values, diffusion_times, ctx):
    amp, D = params
    sqrtbD = np.sqrt(b_values * D)
    s = np.ones_like(sqrtbD)
    nz = sqrtbD != 0
    s[nz] = np.sqrt(np.pi) / (2 * sqrtbD[nz]) * erf(sqrtbD[nz])
    return amp * s

def sphere_stick_signal(params, b_values, diffusion_times, ctx):
    amp, f_s, r, D_intra = params
    sd = ctx.get("small_delta", 3)
    s_sphere = sphere_murdaycotts(r, D_intra, b_values, diffusion_times, sd)
    sqrtbD = np.sqrt(b_values * D_intra)
    s_stick = np.ones_like(sqrtbD)
    nz = sqrtbD != 0
    s_stick[nz] = np.sqrt(np.pi) / (2 * sqrtbD[nz]) * erf(sqrtbD[nz])
    return amp * (f_s * s_sphere + (1 - f_s) * s_stick)

def sphere_stick_sandi_signal(params, b_values, diffusion_times, ctx):
    amp, f_s, r, D_stick = params
    D_soma = 0.5
    sd = ctx.get("small_delta", 3)
    s_sphere = sphere_murdaycotts(r, D_soma, b_values, diffusion_times, sd)
    sqrtbD = np.sqrt(b_values * D_stick)
    s_stick = np.ones_like(sqrtbD)
    nz = sqrtbD != 0
    s_stick[nz] = np.sqrt(np.pi) / (2 * sqrtbD[nz]) * erf(sqrtbD[nz])
    return amp * (f_s * s_sphere + (1 - f_s) * s_stick)

def cylinder_isotropic_signal(params, b_values, diffusion_times, ctx):
    amp, r, D_intra = params
    sd = ctx.get("small_delta", 3)
    n = ctx.get("n_integral_samples", 100)
    mu = np.linspace(-1.0, 1.0, n)
    theta = np.arccos(mu)
    b_perp = np.abs(np.sin(theta)[None, :] ** 2 * b_values[:, None])
    b_para = np.abs(np.cos(theta)[None, :] ** 2 * b_values[:, None])
    dt_all = diffusion_times[:, None] * np.ones(theta.shape)[None, :]
    bp = b_perp.ravel(); ba = b_para.ravel(); dt = dt_all.ravel()
    s_perp = cylinder_perpendicular_signal(r, D_intra, bp, dt, sd)
    g = np.sqrt(ba / (dt - sd / 3.0)) / sd
    s_free = np.exp(-sd**2 * g**2 * D_intra * (dt - sd / 3.0))
    s = s_perp * s_free
    return amp * np.mean(s.reshape(b_perp.shape), axis=1)

def dki_signal(params, b_values, diffusion_times, ctx):
    amp, D, K = params
    mbD = -b_values * D
    return amp * np.exp(mbD + (1.0/6.0) * mbD**2 * K)

def dti_signal(params, b_values, diffusion_times, ctx):
    amp, D = params
    mbD = -b_values * D
    return amp * np.exp(mbD)

def cylinder_sphere_signal(params, b_values, diffusion_times, ctx):
    amp, f_s, r_s, r_c, D_intra = params
    sd = ctx.get("small_delta", 3)
    s_sphere = sphere_murdaycotts(r_s, D_intra, b_values, diffusion_times, sd)
    s_cyl = cylinder_isotropic_signal([1, r_c, D_intra], b_values, diffusion_times, ctx)
    return amp * (f_s * s_sphere + (1 - f_s) * s_cyl)

@dataclass
class ModelSpec:
    name: str
    param_names: Sequence[str]
    param_units: Sequence[str]
    param_names_latex: Sequence[str]
    param_units_latex: Sequence[str]
    initial_guess: Sequence[float]
    bounds: Sequence[tuple]
    signal_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]], np.ndarray]
    # optional options/context
    options: Dict[str, Any] = None
    label: str = None
    label_short: str = None

stick_spec = ModelSpec(
    name="stick",
    param_names=["amp","D_intra"],
    param_units=["a.u.","µm²/ms"],
    param_names_latex=["S_0",r"D_\mathrm{intra}"],
    param_units_latex=["a.u.",r"\micro\metre\squared\per\milli\second"],
    initial_guess=[1.0, 0.5],
    bounds=[(0.1, 1.5), (0.001, 2.0)],
    signal_fn=sticks_signal,
    options=dict(per_diffusion=True),
    label="randomly oriented sticks",
    label_short="ROS"
)

sphere_stick_spec = ModelSpec(
    name="sphere_stick",
    param_names=["amp","f_s","r","D_intra"],
    param_units=["a.u.","","µm","µm²/ms"],
    param_names_latex=["S_0",r"f_\mathrm{S}",r"r",r"D_\mathrm{intra}"],
    param_units_latex=["a.u.","",r"\micro\metre",r"\micro\metre\squared\per\milli\second"],
    initial_guess=[1.0, 0.2, 3.0, 0.5],
    bounds=[(0.01, 1.5), (0.0001, 1.0), (0.01, 20.0), (0.01, 1.5)],
    signal_fn=sphere_stick_signal,
    options=dict(per_diffusion=False),
    label="shpere + randomly oriented sticks",
    label_short="Sphere + ROS"
)

sphere_stick_sandi_spec = ModelSpec(
    name="sphere_stick_sandi",
    param_names=["amp","f_s","r","D_stick"],
    param_units=["a.u.","","µm","µm²/ms"],
    param_names_latex=["S_0",r"f_\mathrm{S}",r"r",r"D_\mathrm{stick}"],
    param_units_latex=["a.u.","",r"\micro\metre",r"\micro\metre\squared\per\milli\second"],
    initial_guess=[1.0, 0.2, 10.0, 0.5],
    bounds=[(0.01, 1.5), (0.0001, 1.0), (0.01, 20.0), (0.01, 1.5)],
    signal_fn=sphere_stick_signal,
    options=dict(per_diffusion=False),
    label="shpere + randomly oriented sticks with fixed soma diffusivity (SANDI style)",
    label_short="sphere + ROS (SANDI style)"
)

cylinder_spec = ModelSpec(
    name="cylinder",
    param_names=["amp","r","D_intra"],
    param_units=["a.u.","µm","µm²/ms"],
    param_names_latex=["S_0",r"r",r"D_\mathrm{intra}"],
    param_units_latex=["a.u.",r"\micro\metre",r"\micro\metre\squared\per\milli\second"],
    initial_guess=[1.0, 3.0, 0.5],
    bounds=[(0.0, 1.5), (0.001, 5.0), (0.001, 1.2)],
    signal_fn=cylinder_isotropic_signal,
    options=dict(n_integral_samples=100,per_diffusion=False),
    label = "randomly oriented cylinders",
    label_short="ROC"
)

cylinder_sphere_spec = ModelSpec(
    name="cylinder_sphere",
    param_names=["amp","f_s","r_s","r_c","D_intra"],
    param_units=["a.u.","","µm","µm","µm²/ms"],
    param_names_latex=["S_0",r"f_\mathrm{S}",r"r_\mathrm{S}",r"r_\mathrm{C}",r"D_\mathrm{intra}"],
    param_units_latex=["a.u.","",r"\micro\metre",r"\micro\metre",r"\micro\metre\squared\per\milli\second"],
    initial_guess=[1.0, 0.1, 15.0, 2.0, 0.5],
    bounds=[(0.01, 5.0),(0.0,1.0),(0.001,50.0),(1e-18,10.0),(0.001,2.5)],
    signal_fn=cylinder_sphere_signal,
    options=dict(n_integral_samples=100,per_diffusion=False),
    label="shpere + randomly oriented cylinders",
    label_short="Sphere + ROC"
)

dki_spec = ModelSpec(
    name="dki",
    param_names=["amp","D","K"],
    param_units=["a.u.","µm²/ms",""],
    param_names_latex=["S_0","D","K"],
    param_units_latex=["a.u.",r"\micro\metre\squared\per\milli\second",""],
    initial_guess=[1.0, 0.1, 0.5],
    bounds=[(0.1,1.5),(0.001,2.0),(0.,4.0)],
    signal_fn=dki_signal,
    options=dict(per_diffusion=True,b_max=15),
    label="diffusion-kurtosis imaging",
    label_short="DKI"
)

dti_spec = ModelSpec(
    name="dti",
    param_names=["amp","D"],
    param_units=["a.u.","µm²/ms"],
    param_names_latex=["S_0","D"],
    param_units_latex=["a.u.",r"\micro\metre\squared\per\milli\second"],
    initial_guess=[1.0, 0.1],
    bounds=[(0.1,1.5),(0.001,2.0)],
    signal_fn=dti_signal,
    options=dict(per_diffusion=True,b_max=4),
    label="diffusion-tensor imaging",
    label_short="DTI"
)

def normalize_by_b(signal, b_values, diffusion_times, normalization_index=0):
    signal = signal.copy()
    unique_dt = np.unique(diffusion_times)
    unique_b = np.unique(b_values)
    norm_b = unique_b[normalization_index]
    for dt in unique_dt:
        idx = np.where(diffusion_times == dt)[0]
        # pick index of b closest to normalization b for these indices
        local_b = b_values[idx]
        i_norm = np.argmin((local_b - norm_b)**2)
        norm_val = signal[idx][i_norm]
        if norm_val != 0:
            signal[idx] /= norm_val
    return signal

def fit_success_boundary_check(result, bounds, initial_guess):
    if not result.success:
        return False
    # fail if any parameter hits upper bound
    for i, (lo, hi) in enumerate(bounds):
        if np.isclose(result.x[i], hi):
            return False
    # fail if optimizer returned initial guess unchanged
    if np.allclose(result.x, np.array(initial_guess)):
        return False
    return True

def keep_mask(dataset, metab, n):
    """Boolean mask of datapoints to keep for a metabolite (True = keep).
    Returns all-True when no outlier detection has been run."""
    outlier = getattr(dataset, "outlier_mask", {}).get(metab, np.zeros(n, dtype=bool))
    return ~outlier

def fit_model_over_metabolites(dataset,
                               spec: ModelSpec,
                               normalize=True,
                               normalization_index=0,
                               per_diffusion=False,
                               initial_guess=None,
                               verbose=True):
    if verbose:
        print("_____________________________")
        print("Fitting "+spec.label)

    results = {}
    b_all = dataset.all_b_values
    diffusion_times = dataset.all_diffusion_times

    if 'b_max' in spec.options.keys():
        diffusion_times = diffusion_times[b_all<=spec.options['b_max']]
        b_cut = b_all[b_all <= spec.options['b_max']]
    else:
        b_cut = b_all

    # optional context passed to signal_fn
    ctx = dict(
        small_delta=dataset.small_delta if hasattr(dataset, "small_delta") else 3,
        n_integral_samples=spec.options.get("n_integral_samples", 100) if spec.options else 100
    )

    if initial_guess is None:
        initial_guess = spec.initial_guess
    if verbose:
        print("_")
        print("Initial guess:")
        for i, parameter in enumerate(spec.param_names):
            print(f"{parameter}: {initial_guess[i]} {spec.param_units[i]}")

    for metab in list(dataset.metabolites):  # copy to allow removal
        keep = keep_mask(dataset, metab, len(b_all))  # exclude flagged outliers
        if 'b_max' in spec.options.keys():
            bmax_sel = b_all <= spec.options['b_max']
        else:
            bmax_sel = np.ones(len(b_all), dtype=bool)
        if not per_diffusion:
            # one fit across all diffusion times
            sel = bmax_sel & keep
            b_used = b_all[sel]
            dt_used = dataset.all_diffusion_times[sel]
            signal_used = dataset.signal[metab][sel]
            def objective(params):
                raw = spec.signal_fn(params, b_used, dt_used, ctx)
                sig = normalize_by_b(raw, b_used, dt_used, normalization_index) if normalize else raw
                residuals = signal_used - sig
                return 0.5 * np.sum(residuals**2)

            res = minimize(objective, initial_guess, method='L-BFGS-B', bounds=spec.bounds)
            print("Warning: Disabled boundary check.")
            #res.success = fit_success_boundary_check(res, spec.bounds, initial_guess)
            results[metab] = res if res.success else None
        else:
            # separate fits per diffusion time (e.g., stick, DKI)
            results[metab] = {}
            for diffusion in dataset.diffusion_times:
                mask_dt = (dataset.all_diffusion_times == diffusion) & keep
                b_this = dataset.all_b_values[mask_dt]
                dt_array = np.ones_like(b_this) * (diffusion + dataset.diffusion_time_increment)
                def objective(params):
                    raw = spec.signal_fn(params, b_this, dt_array, ctx)
                    sig = normalize_by_b(raw, b_this, dt_array, normalization_index) if normalize else raw
                    data = dataset.signal[metab][mask_dt]

                    residuals = data - sig
                    return 0.5 * np.sum(residuals ** 2)

                res = minimize(objective, initial_guess, method='L-BFGS-B', bounds=spec.bounds)
                #res.success = fit_success_boundary_check(res, spec.bounds, initial_guess)
                results[metab][diffusion] = res if res.success else None
    return results


class DMRSModel:
    """A class to fit diffusion models to DMRS datasets."""
    def __init__(self, dmrs_dataset=None):
        self.dataset = dmrs_dataset
        self.results = {}
        self.model_specs = {
            "stick": stick_spec,
            "sphere_stick": sphere_stick_spec,
            "sphere_stick_sandi": sphere_stick_sandi_spec,
            "cylinder": cylinder_spec,
            "cylinder_sphere": cylinder_sphere_spec,
            "dki": dki_spec,
            "dti": dti_spec,
        }
        self.small_delta = 3
        self.diffusion_time_increment = 5 # ms, time to add to mixing time to obtain diffusion time

    def apply_model(self, model, initial_guess=None,
                    normalization_index=0, exclude_Mac=True,
                    modified_spec=None):
        if exclude_Mac and 'Mac' in self.dataset.metabolites:
            self.dataset.metabolites.remove('Mac')
        if modified_spec is None:
            spec = self.model_specs[model]
        else:
            spec = modified_spec
        if initial_guess is None:
            initial_guess = spec.initial_guess

        self.results[model] = fit_model_over_metabolites(
            self.dataset, spec, normalize=True,
            normalization_index=normalization_index,
            per_diffusion=spec.options['per_diffusion'],
            initial_guess=initial_guess
        )
        for metab in self.dataset.metabolites:
            if spec.options['per_diffusion']:
                for diffusion in self.dataset.diffusion_times:
                    if self.results[model][metab][diffusion] is not None:
                        self.results[model][metab][diffusion].sig = None
                    else:
                        print(f"Warning: fit failed for {model} / {metab} @ diffusion time {diffusion} ms, skipping.")
            else:
                if self.results[model][metab] is not None:
                    self.results[model][metab].sig = None
                else:
                    print(f"Warning: fit failed for {model} / {metab}, skipping.")

    def print_results(self, model):
        spec = self.model_specs[model]
        print("__")
        print(f"Results for {model}:")
        for metab in self.dataset.metabolites:
            print(f"{metab}")
            if spec.options['per_diffusion']:
                for j,diffusion in enumerate(self.dataset.diffusion_times):
                    print("_")
                    print(f"Diffusion time: {diffusion} ms")
                    if self.results[model][metab][diffusion] is None:
                        print("fit failed")
                        continue
                    for i, parameter in enumerate(spec.param_names):
                        print(f"{parameter}: {self.results[model][metab][diffusion].x[i]} {spec.param_units[i]}")
            else:
                print("_")
                if self.results[model][metab] is None:
                    print("fit failed")
                    continue
                for i, parameter in enumerate(spec.param_names):
                    if self.results[model][metab].sig is None:
                        print(f"{parameter}: {self.results[model][metab].x[i]} {spec.param_units[i]}")
                    else:
                        print(f"{parameter}: {np.round(self.results[model][metab].x[i],3)} +- {np.round(self.results[model][metab].sig[i],3)}  {spec.param_units[i]}")

    def add_data(self, dmrs_dataset):
        """
        Adds DMRS dataset to the model.
        """
        if isinstance(dmrs_dataset, DMRSDataset):
            self.dataset = dmrs_dataset
            self.dataset.normalize_signal(lowest=True, index=1)
        else:
            print("Error: Could not initialize, unknown DMRS dataset type.")

    def calculate_uncertainties(
            self,
            model,
            initial_guess=None,
            monte_carlo_draws=100,
            modified_spec=None
    ):
        b_all = self.dataset.all_b_values
        diffusion_times = self.dataset.all_diffusion_times

        if modified_spec is None:
            spec = self.model_specs[model]
        else:
            spec = modified_spec

        if initial_guess is None:
            initial_guess = spec.initial_guess
        # optional context passed to signal_fn
        ctx = dict(
            small_delta=self.dataset.small_delta if hasattr(self.dataset, "small_delta") else 3,
            n_integral_samples=spec.options.get("n_integral_samples", 100) if spec.options else 100
        )
        for metab in list(self.dataset.metabolites):  # copy to allow removal
            keep = keep_mask(self.dataset, metab, len(b_all))  # exclude flagged outliers
            if not spec.options['per_diffusion']:
                if self.results[model][metab] is None:
                    continue  # fit failed for this metabolite, nothing to bootstrap
                # one fit across all diffusion times
                raw = spec.signal_fn(self.results[model][metab].x, b_all, diffusion_times, ctx)
                sig = normalize_by_b(raw, b_all, diffusion_times, normalization_index)
                residuals = self.dataset.signal[metab] - sig
                """if 'b_max' in spec.options.keys():
                    residuals = residuals[b_all <= spec.options['b_max']]
                    raw = raw[b_all <= spec.options['b_max']]
                    sig = sig[b_all <= spec.options['b_max']]
                    b_all=b_all[b_all <= spec.options['b_max']]"""
                # only resample residuals among kept, non-zero-b points so an outlier's
                # residual is never shuffled onto a good point
                pool = (b_all != 0) & keep
                params_mc = []
                for i in range(monte_carlo_draws):
                    residuals_pool = residuals[pool]
                    np.random.shuffle(residuals_pool)
                    residuals[pool] = residuals_pool
                    this_signal = sig + residuals
                    this_dataset = copy.deepcopy(self.dataset)
                    this_dataset.metabolites = [metab]
                    this_dataset.signal[metab] = this_signal
                    this_result = fit_model_over_metabolites(
                        this_dataset, spec, normalize=True,
                        normalization_index=normalization_index,
                        per_diffusion=spec.options['per_diffusion'],
                        initial_guess=initial_guess,
                        verbose=False
                    )
                    if this_result[metab] is None:
                        this_result = fit_model_over_metabolites(
                            this_dataset, spec, normalize=True,
                            normalization_index=normalization_index,
                            per_diffusion=spec.options['per_diffusion'],
                            initial_guess=initial_guess,
                            verbose=False
                        )
                    if this_result[metab] is not None:
                        if this_result[metab].success:
                            params_mc.append(this_result[metab].x)

                if len(params_mc) > 0:
                    params_mc = np.array(params_mc)
                    self.results[model][metab].sig = np.std(params_mc, axis=0)
                else:
                    self.results[model][metab].sig = None

            if spec.options['per_diffusion']:
                # one fit across per diffusion times
                for diffusion in self.dataset.diffusion_times:
                    if self.results[model][metab][diffusion] is None:
                        continue  # fit failed for this diffusion time, nothing to bootstrap
                    dsel = (diffusion_times == diffusion)
                    this_diffusion_times = diffusion_times[dsel]
                    this_b_all = b_all[dsel]
                    this_keep = keep[dsel]
                    raw = spec.signal_fn(self.results[model][metab][diffusion].x, this_b_all, this_diffusion_times, ctx)
                    sig = normalize_by_b(raw, this_b_all, this_diffusion_times, normalization_index)
                    residuals = self.dataset.signal[metab][dsel] - sig
                    if 'b_max' in spec.options.keys():
                        bm = this_b_all <= spec.options['b_max']
                        residuals = residuals[bm]
                        sig = sig[bm]
                        this_b_all = this_b_all[bm]
                        this_keep = this_keep[bm]
                    # only resample residuals among kept, non-zero-b points
                    pool = (this_b_all != 0) & this_keep
                    params_mc = []
                    for i in range(monte_carlo_draws):
                        residuals_pool = residuals[pool]
                        np.random.shuffle(residuals_pool)
                        residuals[pool] = residuals_pool
                        this_signal = sig + residuals
                        this_dataset = copy.deepcopy(self.dataset)
                        this_dataset.metabolites = [metab]
                        this_dataset.signal[metab][this_dataset.all_diffusion_times==diffusion][:this_signal.shape[0]] = this_signal
                        this_result = fit_model_over_metabolites(
                            this_dataset, spec, normalize=True,
                            normalization_index=normalization_index,
                            per_diffusion=spec.options['per_diffusion'],
                            initial_guess=initial_guess,
                            verbose=False
                        )
                        if this_result[metab][diffusion] is not None and this_result[metab][diffusion].success:
                            params_mc.append(this_result[metab][diffusion].x)

                    if len(params_mc) > 0:
                        params_mc = np.array(params_mc)
                        self.results[model][metab][diffusion].sig = np.std(params_mc, axis=0)
                    else:
                        self.results[model][metab][diffusion].sig = None

    def plot_results(self, model, store_dir, draft_mode = False, info=None):
        spec = self.model_specs[model]

        ctx = dict(
            small_delta=self.dataset.small_delta if hasattr(self.dataset, "small_delta") else 3,
            n_integral_samples=spec.options.get("n_integral_samples", 100) if spec.options else 100
        )

        if draft_mode:
            dpi = 100
        else:
            dpi = 300

        b_values = self.dataset.b_values
        b_values_conti = np.arange(b_values.min(), b_values.max() + 0.1, 0.1)
        plt.figure(figsize=(6, 6))
        for metab in self.dataset.metabolites:
            # do one plot legend for models with time dependence
            if not spec.options['per_diffusion'] and self.results[model][metab] is not None:
                results_legend = []
                for i, param_name in enumerate(spec.param_names_latex):
                    result_legend = r"$"+ param_name+ r" = "+ str(np.round(self.results[model][metab].x[i],3))
                    if self.results[model][metab].sig is not None:
                        result_legend +=  r"\pm" +  str(np.round(self.results[model][metab].sig[i],3))
                    result_legend += r"\, \mathrm{" + spec.param_units[i]+ r"}$,"
                    results_legend.append(result_legend)

                legend = "\n".join(results_legend[1:])
                plt.plot(
                    [0],
                    [0],
                    label=legend,
                    ls='-',
                    color="black"
                )

            keep = keep_mask(self.dataset, metab, len(self.dataset.all_b_values))
            outlier_label_used = False
            for i,diffusion_time in enumerate(self.dataset.diffusion_times):

                indices = np.where(self.dataset.all_diffusion_times == diffusion_time)[0]
                plt.errorbar(
                    self.dataset.all_b_values[indices],
                    self.dataset.signal[metab][indices],
                    self.dataset.signal_uncertainty[metab][indices],
                    label=(r"data for $\Delta = " + str(diffusion_time) + r"\,\mathrm{ms}$"),
                    ls = 'None',
                    marker = 'x',
                    color = f'C{i}'
                )

                # mark datapoints excluded as outliers
                out_idx = indices[~keep[indices]]
                if len(out_idx) > 0:
                    plt.plot(
                        self.dataset.all_b_values[out_idx],
                        self.dataset.signal[metab][out_idx],
                        ls='None', marker='o', mfc='none', mec='red', ms=10, mew=1.5,
                        label=("excluded outlier" if not outlier_label_used else None),
                    )
                    outlier_label_used = True

                if spec.options['per_diffusion']:
                    if self.results[model][metab][diffusion_time] is None:
                        continue  # fit failed for this diffusion time, skip the model curve
                    params = self.results[model][metab][diffusion_time].x
                    results_legend = [
                        r"$" + param_name + r" = "
                        + str(np.round(self.results[model][metab][diffusion_time].x[i], 3))
                        + r"\pm"
                        + str(0)
                        + r"\, \mathrm{" + spec.param_units[i]
                        + r"}$,"
                        for i, param_name in enumerate(spec.param_names_latex)
                    ]
                    legend = "\n".join(results_legend[1:])
                else:
                    if self.results[model][metab] is None:
                        continue  # fit failed for this metabolite, skip the model curve
                    params = self.results[model][metab].x

                diffusion_times = np.array([diffusion_time]*b_values_conti.shape[0])
                raw = spec.signal_fn(params, b_values_conti, diffusion_times, ctx)
                sig = normalize_by_b(raw, b_values_conti, diffusion_times, normalization_index)

                if spec.options['per_diffusion']:
                    plt.plot(
                        b_values_conti,
                        sig,
                        ls = '-',
                        color = f'C{i}',
                        label=legend
                    )
                else:
                    plt.plot(
                        b_values_conti,
                        sig,
                        ls='-',
                        color=f'C{i}',
                    )

            plt.legend()
            title = f"{metab} dMRS data with {spec.label_short} fit"
            if info:
                title = f"{info} - {title}"
            plt.title(title)
            plt.xlabel('$b$ [ms / µm²]')
            plt.ylabel(r'$S/S(b_{\mathrm{min}})$')
            plt.yscale('log')
            plt.tight_layout()

            storepath = Path(Path(store_dir) /
                 (metab + "_" + model+ ".png"))
            storepath.parent.mkdir(parents=True, exist_ok=True)

            plt.savefig(storepath,dpi=dpi)
            plt.show()
            plt.close()

    def export_csvs(self, model, store_dir):
        csv_path = Path(store_dir)
        Path(csv_path).mkdir(parents=True, exist_ok=True)
        b_values = self.dataset.b_values
        b_values_conti = np.arange(b_values.min(), b_values.max() + 0.01, 0.01)

        spec = self.model_specs[model]

        ctx = dict(
            small_delta=self.dataset.small_delta if hasattr(self.dataset, "small_delta") else 3,
            n_integral_samples=spec.options.get("n_integral_samples", 100) if spec.options else 100
        )

        #export model results
        for metab in self.dataset.metabolites:
            if spec.options['per_diffusion']:
                for diffusion_time in self.dataset.diffusion_times:
                    if self.results[model][metab][diffusion_time] is not None:
                        params = np.array(self.results[model][metab][diffusion_time].x)
                        params_sig = np.array(self.results[model][metab][diffusion_time].sig)
                        df = pd.DataFrame(np.hstack([params,params_sig]).reshape(2,-1), columns=[spec.param_names])
                        df.to_csv((Path(csv_path) / f"fit_parameters_{metab}_{model}_diffusion_time_{int(diffusion_time)}.csv"), index=False)

            else:
                if self.results[model][metab] is not None:
                    params = np.array(self.results[model][metab].x)
                    params_sig = np.array(self.results[model][metab].sig)
                    df = pd.DataFrame(np.hstack([params,params_sig]).reshape(2,-1), columns=[spec.param_names])
                    df.to_csv((Path(csv_path) / f"fit_parameters_{metab}_{model}.csv"), index=False)

        # export model predicitons to csv
        for diffusion_time in self.dataset.diffusion_times:
            df_b_values = pd.DataFrame(b_values_conti, columns=['b-value'])
            dfs_metabs = []

            for metab in self.dataset.metabolites:
                if spec.options['per_diffusion']:
                    res_entry = self.results[model][metab][diffusion_time]
                else:
                    res_entry = self.results[model][metab]
                if res_entry is not None:
                    params = res_entry.x

                    diffusion_times = np.array(
                        [diffusion_time + self.diffusion_time_increment] * b_values_conti.shape[0])
                    raw = spec.signal_fn(params, b_values_conti, diffusion_times, ctx)
                    sig = normalize_by_b(raw, b_values_conti, diffusion_times, normalization_index)

                    dfs_metabs.append(pd.DataFrame(
                        np.round(sig,5),
                        columns=[metab]
                    ))

            df = pd.concat([df_b_values] + dfs_metabs, axis=1)
            df.to_csv((Path(csv_path) / f"fits_{model}_diffusion_time_{int(diffusion_time)}.csv"), index=False)