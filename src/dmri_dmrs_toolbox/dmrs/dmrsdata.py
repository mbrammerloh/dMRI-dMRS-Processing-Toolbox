import re
from codecs import ignore_errors
from dmri_dmrs_toolbox.misc.custom_functions import read_header_file_info
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path



gyromagnetic_ratio = 2.6752218708e8 # 1/sT

class DMRSSequenceData:
    """
    A class that stores dMRS data obtained from LC model fitting.
    """
    def __init__(self,path = None):
        self.metabolite_combinations = dict()
        #self.metabolite_combinations['tNAA'] = ['NAA', 'NAAG']
        #self.metabolite_combinations['tCr'] = ['Cr', 'PCr']
        #self.metabolite_combinations['tCho'] = ['GPC', 'PCho']
        self.metabolite_combinations['Neuronal'] = ['NAA+NAAG', 'Glu']
        self.metabolite_combinations['Glial'] = ['Ins', 'GPC+PCho', 'Gln']
        if path is None:
            self.data = None
        else:
            self.load_data(path)
        diffusion_time  = 0 # ms
        b_value         = 0 # ms / µm^2



    def load_data(self,path):
        """
        Loads dMRS data from file.
        """
        if '.coord' in str(path):
            self.load_lcm_data(path)
        else:
            print("Error: file format not yet supported.")

    def load_lcm_data(self,path):
        """Loads LC model fitting results from .coord file given by path.
        path: .coord file path."""
        series_type = None

        self.data = {}
        metabolites_with_spectra = []
        with open(path) as f:
            vals = []

            for line in f:
                prev_series_type = series_type
                if re.match(".*[0-9]+ points on ppm-axis = NY.*", line):
                    series_type = "ppm"
                elif re.match(".*NY phased data points follow.*", line):
                    series_type = "data"
                elif re.match(".*NY points of the fit to the data follow.*", line):
                    series_type = "completeFit"
                    # completeFit implies baseline+fit
                elif re.match(".*NY background values follow.*", line):
                    series_type = "baseline"
                elif re.match(".*lines in following.*", line):
                    # pattern = r"^\s*(\d+)"
                    # number_of_lines = int(re.search(pattern,line).group(1))
                    # line_number = 0
                    word_pattern = r"lines in following (\w+)"
                    series_type = re.search(word_pattern, line).group(1)
                    read_table = True
                elif re.match("[ ]+[a-zA-Z0-9]+[ ]+Conc. = [-+.E0-9]+$", line):
                    metab = (re.match("[ ]+([a-zA-Z0-9]+)[ ]+Conc. = ([-+.E0-9]+)$", line)).groups(0)[0]
                    metabolites_with_spectra.append(metab)
                    series_type = f'{metab} spectrum'

                if prev_series_type != series_type:  # start/end of chunk...
                    if len(vals) > 0:
                        if read_table and prev_series_type == 'concentration':
                            # Regular expression pattern to match the data in each line
                            pattern = r"\s*(\S+)\s+(\S+)\s+(\S+)\s+(\S+)"

                            # Extracting information into a list of dicts
                            data = []
                            for line in vals[1:]:  # Skip the header line
                                match = re.match(pattern, line)
                                if match:
                                    conc, sd, cr_pc, metabolite = match.groups()
                                    data.append({
                                        'Conc.': conc,
                                        '%SD': sd.rstrip('%'),
                                        '/Cr+PCr': cr_pc,
                                        'Metabolite': metabolite
                                    })

                            # Creating a DataFrame
                            df = pd.DataFrame(data)  # .asytype('float')
                            df['Conc.'] = df['Conc.'].astype('float')
                            df['%SD'] = df['%SD'].astype('float')
                            df['/Cr+PCr'] = df['/Cr+PCr'].astype('float')

                            self.data[prev_series_type] = df
                        elif read_table:
                            self.data[prev_series_type] = vals
                        else:
                            self.data[prev_series_type] = np.array(vals)
                        vals = []
                else:
                    if series_type:
                        if read_table:
                            vals.append(line)
                        else:
                            for x in re.finditer(r"([-+.E0-9]+)[ \t]*", line):
                                v = x.group(1)
                                try:
                                    v = float(v)
                                    vals.append(v)
                                except ValueError:
                                    print("Error parsing line: %s" % (line,))
                                    print(v)
        self.data['metabolites'] = list(df['Metabolite'])

        # define some shorthand variables
        self.metabolites = self.data['metabolites']
        self.signal = dict()
        self.signal_rel = dict()
        self.signal_uncertainty = dict()
        self.signal_rel_uncertainty = dict()
        self.signal_uncertainty_inverse = dict()

        for metab in self.metabolites:
            self.signal[metab]                  = np.array(self.data['concentration'].loc[self.data['concentration']['Metabolite'] == metab]['Conc.'])[0]
            self.signal_rel[metab]              = np.array(self.data['concentration'].loc[self.data['concentration']['Metabolite'] == metab]['/Cr+PCr'])[0]
            self.signal_uncertainty[metab]      = np.array(self.data['concentration'].loc[self.data['concentration']['Metabolite'] == metab]['%SD'] * .01 * self.signal[metab])[0]
            self.signal_rel_uncertainty[metab]  = np.array(self.data['concentration'].loc[self.data['concentration']['Metabolite'] == metab]['%SD'] * .01 * self.signal_rel[metab])[0]
            self.signal_uncertainty_inverse[metab]=np.array(self.data['concentration'].loc[self.data['concentration']['Metabolite'] == metab]['%SD'] * .01 * self.signal[metab])[0]

            if self.signal[metab] == 0:
                self.signal_uncertainty[metab] = np.inf
                self.signal_rel_uncertainty[metab] = np.inf
                self.signal_uncertainty_inverse[metab]= 0

        # adjust metabolite combintations to available data
        for metab_combi in self.metabolite_combinations:
            new_metab_list = [] # list to which to add only quantified metabolites
            invalid_metab_list = []
            for metab in self.metabolite_combinations[metab_combi]:
                if metab in self.metabolites:
                    new_metab_list.append(metab)
                else:
                    invalid_metab_list.append(metab)
            if new_metab_list != self.metabolite_combinations[metab_combi]:
                print(f"Adjusted the metabolite combination {metab_combi} to contain only {new_metab_list} instead of {self.metabolite_combinations[metab_combi]}, as {invalid_metab_list} were not quantified.")
            self.metabolite_combinations[metab_combi] = new_metab_list


    def combine_metabolites(self):
        """Calculates common metabolite concentration combinations from data.
        Currently implemented:
        tNAA = NAA + NAAG
        tCr  = Cr + PCr
        tCho = GPC + PCho
        Neuronal = NAA + Glu
        Glial    = Ins + tCho + Gln"""
        for metab_combi in self.metabolite_combinations:
            if metab_combi not in self.metabolites:
                self.metabolites.append(metab_combi)
                self.signal[metab_combi] = np.sum([self.signal[metab] for metab in self.metabolite_combinations[metab_combi]])
                self.signal_rel[metab_combi] = np.sum([self.signal_rel[metab] for metab in self.metabolite_combinations[metab_combi]])
                if self.signal[metab_combi] == 0:
                    self.signal_uncertainty[metab_combi] = np.inf
                    self.signal_rel_uncertainty[metab_combi] = np.inf
                    self.signal_uncertainty_inverse[metab_combi] = 0
                else:
                    self.signal_uncertainty[metab_combi] = np.sum(
                        [self.signal_uncertainty[metab]**2 for metab in self.metabolite_combinations[metab_combi]]
                    )**.5
                    self.signal_rel_uncertainty[metab_combi] = np.sum(
                        [self.signal_rel_uncertainty[metab] ** 2 for metab in self.metabolite_combinations[metab_combi]]
                    ) ** .5
                    self.signal_uncertainty_inverse[metab_combi] = 1/self.signal_uncertainty[metab_combi]


    def add_metabolite_combination(self,combination_name, combined_metabolites):
        """
        Helper function for adding metabolite combinations.
        combination_name: str, name of the metabolite combination (e.g. "tNAA")
        combined_metabolites: list of metabolites to add (e.g. ["NAA", "NAAG"]).
        """
        self.metabolite_combinations[combination_name] = combined_metabolites

class DMRSDataset:
    """Class to store and plot a collection of dMRS sequence data."""
    def __init__(self, data=None):
        self.signal = dict()
        self.signal_rel = dict()
        self.signal_uncertainty = dict()
        self.signal_rel_uncertainty = dict()
        self.signal_uncertainty_inverse = dict()

        # per-metabolite boolean outlier masks (True = excluded from fitting),
        # aligned with all_b_values / all_diffusion_times. Empty => nothing excluded.
        self.outlier_mask = dict()

        self.diffusion_time_increment = 5 # ms

        if data is not None:
            self.add_data(data)
    def add_data(self,list_of_dmrs_data):
        self.data = list_of_dmrs_data
        b_values = []
        for data in self.data:
            b_values.append(data.b_value)
        self.all_b_values = np.array(b_values)
        self.b_values = np.unique(self.all_b_values)

        diffusion_times = []
        for data in self.data:
            diffusion_times.append(data.diffusion_time)
        self.all_diffusion_times = np.array(diffusion_times)
        self.diffusion_times = np.unique(self.all_diffusion_times)
        self.all_mixing_times = self.all_diffusion_times - self.diffusion_time_increment
        self.mixing_times = np.unique(self.all_mixing_times)

        self.metabolites = []
        for i in range(len(self.data)):
            for metab in self.data[i].metabolites:
                skip_metabolite = False
                for j in range(len(self.data)):
                    if metab not in self.data[j].metabolites:
                        #print("Metabolite %s not in all data, ignoring data." % (metab,))
                        skip_metabolite = True
                if not skip_metabolite and metab not in self.metabolites:
                    self.metabolites.append(metab)
        print("Valid metabolites: %s" % self.metabolites)

        # sort datasets by diffusion time and b_value
        combined = list(zip(
            self.data, self.all_diffusion_times,self.all_b_values
        ))
        combined.sort(key=lambda tup: (tup[1], tup[2]))
        self.datasets, self.all_diffusion_times, self.all_b_values = map(
            list, zip(*combined)
        )
        self.all_diffusion_times=np.array(self.all_diffusion_times)
        self.all_b_values = np.array(self.all_b_values)

        # keep self.data aligned (sorted) with the all_* arrays
        self.data = list(self.datasets)

        # Detect genuine duplicate (diffusion time, b-value) acquisitions. Note the
        # grid may be ragged (fewer b-values at longer diffusion times), which is valid.
        pairs = list(zip(self.all_diffusion_times, self.all_b_values))
        if len(set(pairs)) != len(pairs):
            raise ValueError("Duplicate (b-value, diffusion time) combinations detected, "
                             "please review dataset.")

        # Build signal arrays directly from the sorted per-scan datasets so they stay
        # element-for-element aligned with all_b_values / all_diffusion_times even when
        # the acquisition grid is ragged.
        for metab in self.metabolites:
            self.signal[metab]                     = np.array([d.signal[metab] for d in self.data])
            self.signal_rel[metab]                 = np.array([d.signal_rel[metab] for d in self.data])
            self.signal_uncertainty[metab]         = np.array([d.signal_uncertainty[metab] for d in self.data])
            self.signal_rel_uncertainty[metab]     = np.array([d.signal_rel_uncertainty[metab] for d in self.data])
            self.signal_uncertainty_inverse[metab] = np.array([d.signal_uncertainty_inverse[metab] for d in self.data])

    def load_mc_sim_result(self, dwi_path, scheme_path, binary = True):
        """
        Function to load Monte Carlo simulation results for the dMRS signal.
        Current implementation assumes a PGSE acquisiton (Stejskal-Tanner) with multiple directions
        that is avereaged over directions to get an isotropic weighting of directions.
        dwi_path is the path of the DWI.bfloat output file of the MC simulation.
        scheme_path is the path of the scheme file of the MC simulation, needed to extract b-values and diffusion times. So far, only stejskal tanner sequences are implemented
        """
        self.metabolites = ['simulation']
        column = []  # Initialize an empty list to store the values
        if binary:
            lines = np.fromfile(dwi_path, dtype="float32")
            for line in lines:
                column.append(float(line))  # Assuming columns are 0-based
        else:
            try:
                with open(dwi_path, 'r') as file:
                    for line in file:
                        # Split each line into space-separated values and extract
                        columns = line.strip().split()
                        column.append(
                            float(columns[0]))  # Assuming columns are 0-based
            except FileNotFoundError:
                print(f"File not found: {dwi_path}")

        signal = pd.DataFrame(column, columns=['signal'])

        scheme = np.loadtxt(scheme_path, delimiter=" ", skiprows=1)

        direction_x = pd.DataFrame(scheme[:, 0], columns=["direction_x"]) # au
        direction_y = pd.DataFrame(scheme[:, 1], columns=["direction_y"]) # au
        direction_z = pd.DataFrame(scheme[:, 2], columns=["direction_z"]) # au
        gradient = pd.DataFrame(scheme[:, 3], columns=["gradient"]) # T/m
        diffusion_time = pd.DataFrame(scheme[:, 4]*1e3, columns=["diffusion_time"]) # ms
        small_delta =  pd.DataFrame(scheme[:, 5]*1e3, columns=["small_delta"]) # ms
        echo_time = pd.DataFrame(scheme[:, 6] * 1e3, columns=["echo_time"]) # ms

        mc_data = pd.concat([
            signal,
            direction_x,
            direction_y,
            direction_z,
            gradient,
            diffusion_time,
            small_delta,
            echo_time,
        ], axis = 1)

        # here we round because there are some rounding errors in the generation of the scheme file
        mc_data['b-value'] = np.round(
                gyromagnetic_ratio**2 * mc_data['gradient']**2  * mc_data['small_delta']**2 * (mc_data['diffusion_time'] - mc_data['small_delta']/3)
        *1e-18,2)

        # bring data into DMRSDataset convention and averaging over direcitons
        for metab in self.metabolites: # in this case only one
            self.signal[metab] = []
            self.signal_uncertainty[metab] = []
            self.signal_uncertainty_inverse[metab] = []

            b_values = []
            diffusion_times = []
            averaged_signals = []
            averaged_signals_uncertainty = []

            for b_value in np.sort(mc_data['b-value'].unique()):
                this_data = mc_data.loc[mc_data['b-value'] == b_value]
                for delta in np.sort(mc_data['diffusion_time'].unique()):
                    if delta !=505:
                        b_values.append(b_value)
                        diffusion_times.append(delta)
                        averaged_signals.append(
                            this_data['signal'][this_data['diffusion_time'] == delta].mean()
                        )
                        averaged_signals_uncertainty.append(
                            0
                        )
                        self.signal[metab].append(averaged_signals[-1])
                        self.signal_uncertainty[metab].append(averaged_signals_uncertainty[-1])
                        self.signal_uncertainty_inverse[metab].append(1)
            self.signal[metab] = np.array(self.signal[metab])
            self.signal_uncertainty[metab] = np.array(self.signal_uncertainty[metab])
            self.signal_uncertainty_inverse[metab] = np.array(self.signal_uncertainty_inverse[metab])

        self.all_b_values = np.array(b_values)
        self.b_values = np.unique(self.all_b_values)

        self.all_mixing_times = np.array(diffusion_times)-self.diffusion_time_increment
        self.mixing_times = np.unique(self.all_mixing_times)

        self.all_diffusion_times = np.array(diffusion_times)
        self.diffusion_times = np.unique(self.all_diffusion_times)

    def generate_averaged_dataset(self, dmrs_datasets):
        """
        Generates an averaged dMRS dataset by averaging a given set of repetitions
        of a dMRS experiment.
        """
        self.metabolites = dmrs_datasets[0].metabolites
        self.all_b_values = dmrs_datasets[0].all_b_values
        self.b_values = dmrs_datasets[0].b_values

        self.all_mixing_times = dmrs_datasets[0].all_mixing_times
        self.mixing_times = dmrs_datasets[0].mixing_times

        self.all_diffusion_times = dmrs_datasets[0].all_diffusion_times
        self.diffusion_times = dmrs_datasets[0].diffusion_times

        for metab in self.metabolites: # in this case only one
            self.signal[metab] = []
            self.signal_uncertainty[metab] = []
            self.signal_uncertainty_inverse[metab] = []
            averaged_signals = []
            averaged_signals_uncertainty = []
            averaged_signals_uncertainty_inverse = []
            for b_value in self.b_values:
                for delta in self.diffusion_times:
                    signals = []
                    uncertainties = []

                    for dataset in dmrs_datasets:
                        signals.append(
                            dataset.signal[metab][(dataset.all_b_values == b_value)*(dataset.all_diffusion_times == delta)][0]
                        )
                        uncertainties.append(
                            dataset.signal_uncertainty[metab][(dataset.all_b_values == b_value)*(dataset.all_diffusion_times == delta)][0]
                        )
                    averaged_signals.append(np.mean(signals))
                    averaged_signals_uncertainty.append(np.mean(uncertainties)/np.sqrt(len(uncertainties)))
                    if averaged_signals_uncertainty[-1]!=0:
                        averaged_signals_uncertainty_inverse.append(1/averaged_signals_uncertainty[-1])
                    else:
                        averaged_signals_uncertainty_inverse.append(1)

                    self.signal[metab].append(averaged_signals[-1])
                    self.signal_uncertainty[metab].append(averaged_signals_uncertainty[-1])
                    self.signal_uncertainty_inverse[metab].append(averaged_signals_uncertainty_inverse[-1])

            self.signal[metab] = np.array(self.signal[metab])
            self.signal_uncertainty[metab] = np.array(self.signal_uncertainty[metab])
            self.signal_uncertainty_inverse[metab] = np.array(self.signal_uncertainty_inverse[metab])


    def plot(self, diffusion_times = None, metabolites=None):
        if metabolites is None:
            metabolites = self.metabolites
        if diffusion_times is None:
            diffusion_times = self.diffusion_times
        for diffusion_time in diffusion_times:
            if diffusion_time not in self.diffusion_times:
                return print("Invalid diffusion time, we have only ", self.diffusion_times)
        plt.figure(figsize=(6,6))
        for metab in metabolites:
            for diffusion_time in diffusion_times:
                signals = self.signal[metab][np.where(self.all_diffusion_times == diffusion_time)]
                signal_uncertainty = self.signal_uncertainty[metab][np.where(self.all_diffusion_times == diffusion_time)]

                plt.errorbar(
                    self.b_values,
                    signals,
                    signal_uncertainty,
                    linestyle='none', marker='x', label=f'{metab}, {diffusion_time} ms'
                )

        plt.title('dMRS signal decays')
        plt.ylabel('$S$ [a.u.]')
        plt.xlabel('$b$ [ms/µm²]')
        plt.yscale('log')
        plt.legend()
        plt.show()

    def export_csvs(self,csv_path):
        Path(csv_path).mkdir(parents=True, exist_ok=True)
        for diffusion_time in self.diffusion_times:
            df_b_values = pd.DataFrame(self.b_values, columns=['b-value'])
            dfs_metabs = []
            for metab in self.metabolites:
                dfs_metabs.append(pd.DataFrame(np.round(self.signal[metab][np.where(self.all_diffusion_times == diffusion_time)],5), columns = [metab]))
                dfs_metabs.append(
                    pd.DataFrame(np.round(
                        self.signal_uncertainty[metab][np.where(self.all_diffusion_times == diffusion_time)],5
                    ), columns=[metab + "_crlb"]))
            df = pd.concat([df_b_values]+dfs_metabs, axis=1)
            df.to_csv((Path(csv_path) / f"signals_diffusion_time_{diffusion_time}.csv"), index=False)

    def get_signals(self, metabolite, diffusion_times, b_values):
        signals = []
        for dataset in self.data:
            if dataset.diffusion_time in diffusion_times and dataset.b_value in b_values:
                signals.append(dataset.signal[metabolite])
        return np.array(signals)

    def get_signals_rel(self, metabolite, diffusion_times, b_values):
        signals = []
        for dataset in self.data:
            if dataset.diffusion_time in diffusion_times and dataset.b_value in b_values:
                signals.append(dataset.signal[metabolite])
        return np.array(signals)

    def get_signals_uncertainty(self, metabolite, diffusion_times, b_values):
        signals = []
        for dataset in self.data:
            if dataset.diffusion_time in diffusion_times and dataset.b_value in b_values:
                signals.append(dataset.signal_uncertainty[metabolite])
        return np.array(signals)

    def get_signals_rel_uncertainty(self, metabolite, diffusion_times, b_values):
        signals = []
        for dataset in self.data:
            if dataset.diffusion_time in diffusion_times and dataset.b_value in b_values:
                signals.append(dataset.signal_rel_uncertainty[metabolite])
        return np.array(signals)

    def get_signals_uncertainty_inverse(self, metabolite, diffusion_times, b_values):
        signals = []
        for dataset in self.data:
            if dataset.diffusion_time in diffusion_times and dataset.b_value in b_values:
                signals.append(dataset.signal_uncertainty_inverse[metabolite])
        return np.array(signals)

    def get_data(self,diffusion_time,b_value, return_indices=False):
        """
        Returns list with all data matching given diffusion time and b_value.
        If return_indices is True, returns indices of data matching the given diffusion time and b_value.
        """
        datasets = []
        indices = []
        for i,data in enumerate(self.data):
            if data.diffusion_time == diffusion_time and data.b_value == b_value:
                datasets.append(data)
                indices.append(i)
        if return_indices:
            return indices
        else:
            return datasets

    def normalize_signal(self,lowest =True, index=0):
        """
        Normalizes the signals and signal_uncertainties objects of the datasets by the signal at the lowest b_value for a given diffusion time.
        Caution: data objects remain unchanged.
        Lowest: Normalizes by lowest b-value, otherwise index specifies the index of the normalization b-value, where 0 is lowest, 1 is second lowest, etc.
        """
        if lowest:
            normalization_b_value = np.min(self.b_values)
        else:
            normalization_b_value = np.sort(self.b_values)[index]
        print("Normalizing signals and signals_uncertainty by b_value =", normalization_b_value, " ms/µm².")
        
        for diffusion_time in self.diffusion_times:
            for metab in self.metabolites:
                signal_normalization_b_value = self.signal[metab][(self.all_diffusion_times == diffusion_time) *(self.all_b_values == normalization_b_value)][0]
                for b_value in self.b_values:
                    this_index = np.where((self.all_b_values==b_value)*(self.all_diffusion_times==diffusion_time))[0]
                    for i in this_index:
                        this_signal = self.signal[metab][i]
                        this_signal_uncertainty = self.signal_uncertainty[metab][i]
                        this_signal_uncertainty_inverse = self.signal_uncertainty_inverse[metab][i]

                        if signal_normalization_b_value == 0:
                            this_signal_normalized = 0
                            this_signal_uncertainty_normalized = np.inf
                            this_signal_uncertainty_inverse = 0

                        else:
                            this_signal_normalized              = this_signal / signal_normalization_b_value
                            this_signal_uncertainty_normalized  = this_signal_uncertainty / signal_normalization_b_value
                            this_signal_uncertainty_inverse     = this_signal_uncertainty_inverse * signal_normalization_b_value

                        self.signal[metab][i]              = this_signal_normalized
                        self.signal_uncertainty[metab][i]   = this_signal_uncertainty_normalized
                        self.signal_uncertainty_inverse[metab][i] = this_signal_uncertainty_inverse

    def detect_outliers(self, k_sigma=3.0, rel_tol=0.05, min_points=3, verbose=True):
        """
        Detect b-decay outliers per metabolite and per diffusion time.

        The diffusion-weighted signal must decrease monotonically with b-value within a
        diffusion time. A monotonicity violation between an ordered pair b_i < b_j occurs
        when the higher-b point carries more signal than the lower-b point beyond noise:

            e_ij = S_j - S_i - thr_ij  > 0,   thr_ij = k_sigma*sqrt(CRLB_i^2+CRLB_j^2)+rel_tol

        Such a violation means EITHER point i is too low OR point j is too high. To resolve
        this robustly (a single bad point can otherwise corrupt the assessment of its
        neighbours) the outliers are removed greedily: repeatedly score each point by the
        number of violated pairs it participates in (tie-break by summed excess, then by
        larger CRLB) and remove the worst offender, until no violations remain. This flags
        both too-high and too-low points symmetrically.

        Should be called after normalize_signal(). Results are stored in
        self.outlier_mask[metab] (boolean, True = excluded), aligned with all_b_values /
        all_diffusion_times. The normalization point (lowest b of each diffusion time) is
        never removed, and at least `min_points` points are kept per diffusion time.
        """
        self.outlier_mask = {}
        self._outlier_records = []
        for metab in self.metabolites:
            mask = np.zeros(len(self.all_b_values), dtype=bool)
            signal = self.signal[metab]
            crlb = self.signal_uncertainty[metab]
            for diffusion_time in self.diffusion_times:
                idx = np.where(self.all_diffusion_times == diffusion_time)[0]
                # sort this diffusion time's points by ascending b-value
                idx = idx[np.argsort(self.all_b_values[idx])]
                b_norm = self.all_b_values[idx][0]  # normalization (lowest) b-value
                # active set: kept points with finite signal / CRLB, ascending in b
                active = [i for i in idx
                          if np.isfinite(signal[i]) and np.isfinite(crlb[i])]

                while len(active) > min_points:
                    n_viol = {i: 0 for i in active}
                    excess = {i: 0.0 for i in active}
                    any_violation = False
                    for a in range(len(active)):
                        for b in range(a + 1, len(active)):
                            i, j = active[a], active[b]  # b_i < b_j
                            thr = k_sigma * np.sqrt(crlb[i] ** 2 + crlb[j] ** 2) + rel_tol
                            e = signal[j] - signal[i] - thr  # signal should decrease
                            if e > 0:
                                any_violation = True
                                n_viol[i] += 1; n_viol[j] += 1
                                excess[i] += e;  excess[j] += e
                    if not any_violation:
                        break
                    # candidates: violating points, never the normalization reference
                    cands = [i for i in active
                             if n_viol[i] > 0 and self.all_b_values[i] != b_norm]
                    if not cands:
                        break
                    worst = max(cands, key=lambda i: (n_viol[i], excess[i], crlb[i]))
                    mask[worst] = True
                    active.remove(worst)
                    self._outlier_records.append({
                        "Metabolite": metab,
                        "Diffusion Time": diffusion_time,
                        "b-value": self.all_b_values[worst],
                        "signal": signal[worst],
                        "n_violations": n_viol[worst],
                        "excess": excess[worst],
                    })
            self.outlier_mask[metab] = mask

        if verbose:
            n = len(self._outlier_records)
            print(f"Outlier detection (k_sigma={k_sigma}, rel_tol={rel_tol}): "
                  f"flagged {n} datapoint(s).")
            for rec in self._outlier_records:
                print(f"  {rec['Metabolite']:<10s} "
                      f"Delta={rec['Diffusion Time']:.1f} ms  b={rec['b-value']:.3g}  "
                      f"S={rec['signal']:.3f}  n_viol={rec['n_violations']}  "
                      f"excess={rec['excess']:.3f}")

    def export_outliers(self, csv_path):
        """Write the datapoints flagged by detect_outliers() to a CSV for QA."""
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        records = getattr(self, "_outlier_records", [])
        columns = ["Metabolite", "Diffusion Time", "b-value", "signal", "n_violations", "excess"]
        df = pd.DataFrame(records, columns=columns)
        df.to_csv(csv_path, index=False)

    def combine_metabolites(self):
        """Calculates common metabolite concentration combinations from data.
        Currently implemented:
        tNAA = NAA + NAAG
        tCr  = Cr + PCr
        tCho = GPC + PCho
        Neuronal = NAA + Glu
        Glial    = Ins + tCho + Gln"""
        for data in self.data:
            data.combine_metabolites()
        for metab_combi in self.data[0].metabolite_combinations:
            self.metabolites.append(metab_combi)
            self.signal[metab_combi] = []
            self.signal_uncertainty[metab_combi] = []
            self.signal_rel[metab_combi] = []
            self.signal_rel_uncertainty[metab_combi] = []
            self.signal_uncertainty_inverse[metab_combi] = []

            for i, data in enumerate(self.data):
                self.signal[metab_combi].append(data.signal[metab_combi])
                self.signal_uncertainty[metab_combi].append(data.signal_uncertainty[metab_combi])
                self.signal_rel[metab_combi].append(data.signal_rel[metab_combi])
                self.signal_rel_uncertainty[metab_combi].append(data.signal_rel_uncertainty[metab_combi])
                self.signal_uncertainty_inverse[metab_combi].append(data.signal_uncertainty_inverse[metab_combi])

            self.signal[metab_combi] = np.array(self.signal[metab_combi])
            self.signal_uncertainty[metab_combi] = np.array(self.signal_uncertainty[metab_combi])
            self.signal_rel[metab_combi] = np.array(self.signal_rel[metab_combi])
            self.signal_rel_uncertainty[metab_combi] = np.array(self.signal_rel_uncertainty[metab_combi])
            self.signal_uncertainty_inverse[metab_combi] = np.array(self.signal_uncertainty_inverse[metab_combi])

    def remove_diffusion_time(self, diffusion_time):
        return print("not implemented fully!!!")
        diffusion_time_to_remove=diffusion_time
        if diffusion_time not in self.diffusion_times:
            print("Diffusion time not found, nothing to remove")
        else:
            for metab in self.metabolites:
                new_all_b_values = []
                new_all_diffusion_times = []
                new_signals = []
                new_signal_uncertainty = []
                new_signal_uncertainty_inverse = []
                for b_value in self.b_values:
                    for diffusion_time in self.diffusion_times:
                        if diffusion_time!=diffusion_time_to_remove:
                            new_all_b_values.append(b_value)
                            new_all_diffusion_times.append(diffusion_time)
                            new_signals.append(self.signal[metab][(self.all_b_values==b_value)*(self.all_diffusion_times==diffusion_time)])

    def plot_spectra(self, csv_path = None, show_plots = False):
        print("Plotting spectra")

        if csv_path is not None:
            csv_path = Path(csv_path)
            csv_path.mkdir(parents=True, exist_ok=True)

        for i_dt, diffusion_time in enumerate(self.all_diffusion_times):
            complete_data = ""
            for data_part in self.data[i_dt].data['ppm']:
                complete_data = complete_data + data_part
            ppm_axis = np.fromstring(complete_data, sep=' ')

            complete_data = ""
            for data_part in self.data[i_dt].data['data']:
                complete_data = complete_data + data_part
            spectrum = np.fromstring(complete_data, sep=' ')

            plt.plot(ppm_axis, spectrum)
            plt.xlim(ppm_axis[0],ppm_axis[-1])
            if show_plots:
                plt.show()
            plt.close()

            ppm_axis_df = pd.DataFrame(ppm_axis, columns=['ppm'])
            spectrum_df = pd.DataFrame(spectrum, columns=['spectrum'])

            df = pd.concat([ppm_axis_df, spectrum_df], axis=1)

            if csv_path is not None:
                df.to_csv(Path(csv_path) / Path(f"spectrum_TD_{diffusion_time}_b_{self.all_b_values[i_dt]}.csv"), index=False)

    def load_lcm_quantified_directory(self, path_lcm_quantified, path_bruker_rawdata):
        datasets = []
        b_values = []
        path_lcm_quantified = Path(path_lcm_quantified)
        path_bruker_rawdata  = Path(path_bruker_rawdata)
        for file in sorted(Path(path_lcm_quantified).iterdir()):
            if '.coord' in file.name:
                datasets.append(DMRSSequenceData(file))
                seq_no = file.name.split("_data_")[0].split("_")[-1]
                method_path = path_bruker_rawdata/seq_no/"method"
                header_info= read_header_file_info(method_path,["Bvalue", "DiffusionTime"],[])
                b_value = header_info['Bvalue']/1e3
                datasets[-1].b_value = b_value
                datasets[-1].diffusion_time = header_info['DiffusionTime']

        self.add_data(datasets)