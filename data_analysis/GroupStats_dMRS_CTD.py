#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare groups microstrcture
@author: localadmin
"""

import os
import sys
import matplotlib.pyplot as plt
from dmri_dmrs_toolbox.dmrs.dmrsmodel import DMRSModel
import numpy as np

# NOTE: must match the subjects actually fitted by main_dMRS_ctd.py (Step2_fitting)
subj_list = [f'sub-{i:02d}' for i in list(np.arange(3,30))]    # list of subjects to analyse. Excluded atm: 11 15

cfg                         = {}
cfg['subj_list']            = subj_list
cfg['data_path']            = "/media/localadmin/DATA/data/CTD/"          # path to where the data from the cohort is
cfg['prep_foldername']      = 'preprocessed'    # name of the preprocessed folder (keep 'preprocessed' as default)
cfg['analysis_foldername']  = 'analysis'        # name of the analysis folder (keep 'analysis' as default)
cfg['common_folder']        = "/home/localadmin/Software/dMRI-dMRS-Processing-Toolbox/common/"  # path to the common folder with files needed throught the pipeline
cfg['scan_list_name']       = 'ScanList_CTD.xlsx'   # name of the excel file containing the metadata of the cohort
cfg['model_list']           =  ['cylinder', 'dti', 'stick', 'dki', 'sphere_stick','cylinder_sphere']
cfg['metabolites']          = ['NAA+NAAG','Glu','Ins','GPC+PCho','Cr+PCr','Tau','Gln']              # metabolites for analysis


from dmri_dmrs_toolbox.misc.bids_structure import *
from dmri_dmrs_toolbox.misc.custom_functions import *

from scipy.optimize import curve_fit
import numpy as np
import math
import pandas as pd
import glob
import copy
import seaborn as sns
from scipy.stats import ttest_ind, mannwhitneyu
from pathlib import Path

output_folder = Path(cfg['data_path'])/'results'
output_folder.mkdir(parents=True, exist_ok=True)

scan_list   = pd.read_excel(os.path.join(cfg['data_path'], cfg['scan_list_name']))

def cohen_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    pooled_var = ((nx - 1)*np.var(x, ddof=1) + (ny - 1)*np.var(y, ddof=1)) / dof
    if not np.isfinite(pooled_var) or pooled_var <= 0:
        return np.nan
    return (np.mean(x) - np.mean(y)) / np.sqrt(pooled_var)


def bh_fdr(pvals):
    """Benjamini-Hochberg FDR-adjusted p-values. NaNs are passed through and
    excluded from the correction family."""
    p = np.asarray(pvals, dtype=float)
    out = np.full(p.shape, np.nan)
    finite = np.isfinite(p)
    pv = p[finite]
    n = pv.size
    if n == 0:
        return out
    order = np.argsort(pv)
    ranked = pv[order]
    adj = ranked * n / (np.arange(n) + 1)
    # enforce monotonicity from the largest p-value downwards, then clip to [0, 1]
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    res = np.empty(n)
    res[order] = adj
    out[finite] = res
    return out



######## MODEL-WISE OPERATIONS ########
dmrsmodel = DMRSModel()

stats_records = []  # collect group-comparison statistics across all models

for model in cfg['model_list']:
    all_diffusion_times = []
    print(f'Working with {model}...')
    specs = dmrsmodel.model_specs[model]
    params = specs.param_names
    nb_params = len(params)

    # create one dataframe for fitting results for each diffusion time
    df_all_data = pd.DataFrame()

    for subj in cfg['subj_list']:
        print('Working on subject ' + subj + '...')
        # Extract data for subject
        subj_data = scan_list[(scan_list['study_name'] == subj)].reset_index(drop=True)

        group = subj_data['group'].iloc[0]
        if '?' in group:
            group = 'no group yet'

        # List of acquisition sessions
        sess_list = [x for x in list(subj_data['sessNo'].unique()) if not math.isnan(x)]  # clean NaNs

        ######## SESSION-WISE OPERATIONS ########
        for sess in sess_list:
            print('Working on session ' + str(sess) + '...')

            data_path = cfg['data_path']

            bids_strc_analysis = create_bids_structure(subj=subj, sess=sess, datatype='dmrs', root=data_path,
                                                       folderlevel='derivatives',
                                                       workingdir=cfg['analysis_foldername'],
                                                       description=model)

            # Obtain diffusion times
            diffusion_times = ["all"]
            if 'per_diffusion' in specs.options:
                if specs.options['per_diffusion']:
                    csv_dir = Path(bids_strc_analysis.get_path()) / 'csvs'
                    diffusion_times = []
                    if csv_dir.is_dir():
                        for csv_file in csv_dir.iterdir():
                            diffusion_times.append(csv_file.name.split(".csv")[0].split("_")[-1])
                        diffusion_times = list(dict.fromkeys(diffusion_times))
                        diffusion_times = np.array(diffusion_times).astype("int")
                        diffusion_times.sort()
            all_diffusion_times = np.hstack([all_diffusion_times, diffusion_times])

            for metab in cfg['metabolites']:
                for dt in diffusion_times:
                    if dt != "all":
                        fit_params_path = Path(
                            bids_strc_analysis.get_path()) / "csvs" / f'fit_parameters_{metab}_{model}_diffusion_time_{dt}.csv'
                    else:
                        fit_params_path = Path(
                            bids_strc_analysis.get_path()) / "csvs" / f'fit_parameters_{metab}_{model}.csv'
                    if fit_params_path.is_file():
                        fit_parameters = pd.read_csv(fit_params_path)
                    else:
                        fit_parameters = pd.DataFrame(
                            np.full((2, nb_params), np.nan),
                            columns=params
                        )
                        group = "no data"
                    fit_parameters['Metabolite'] = metab
                    fit_parameters["Subject"] = subj
                    fit_parameters["Group"] = group
                    fit_parameters["Session"] = int(sess)
                    fit_parameters["Diffusion Time"] = dt

                    df_all_data = pd.concat([df_all_data, fit_parameters], ignore_index=True)

    ######## FOR ALL THE DATA OF THIS MODEL ########
    all_diffusion_times = list(dict.fromkeys(all_diffusion_times))
    all_diffusion_times.sort()
    for dt in all_diffusion_times:
        df_all_data_dt = df_all_data.loc[df_all_data['Diffusion Time'] == dt]

        plot_rows = []

        for _, row in df_all_data_dt.iloc[::2].iterrows(): # only take mean values for now
            metab = row['Metabolite']
            group = row['Group']
            session = row['Session']
            subject = row['Subject']

            for param in params:
                values = np.ravel(row[param])
                for val in values:
                    plot_rows.append({
                        "Parameter": param,
                        "Metabolite": metab,
                        "Group": group,
                        "Session": session,
                        "Subject": subject,
                        "Model": model,
                        "Value": float(val)
                    })
        df_long = pd.DataFrame(plot_rows)

        ## Create labels
        group_counts = df_long[['Group', 'Subject']].drop_duplicates().groupby('Group').count()['Subject']

        desired_order = ['WT', 'KI', 'HTZ', 'no group yet', 'no data']
        HUE_ORDER = [g for g in desired_order if g in group_counts.index]

        df_long['Group'] = (df_long['Group']
                            .astype(str)
                            .str.strip())
        df_long['Group'] = pd.Categorical(df_long['Group'], categories=desired_order, ordered=True)
        df_long["Group"] = df_long["Group"].astype(str).str.strip()
        df_long = df_long[df_long["Group"].isin(HUE_ORDER)].copy()
        df_long["Group"] = pd.Categorical(df_long["Group"], categories=HUE_ORDER, ordered=True)

        ## Plot
        print('Plotting')
        amp_idx = params.index("amp")
        plotting_params = params[:amp_idx] + params[amp_idx+1:]

        lims = dmrsmodel.model_specs[model].bounds
        plotting_lims = lims[:amp_idx] + lims[amp_idx+1:]

        plotting_labels = dmrsmodel.model_specs[model].param_names_latex
        plotting_labels = plotting_labels[:amp_idx] + plotting_labels[amp_idx+1:]

        plotting_units = dmrsmodel.model_specs[model].param_units
        plotting_units = plotting_units[:amp_idx] + plotting_units[amp_idx+1:]

        fig, axes = plt.subplots(len(plotting_params), 1, figsize=(10, 10))
        fig.subplots_adjust(wspace=0.05, hspace=0.11, top=0.95, bottom=0.1, left=0.05, right=0.95)

        if len(plotting_params) == 1:
            axes = [axes]

        for i, (ax, param, lim, plotting_label, plotting_unit) in enumerate(zip(axes, plotting_params,plotting_lims, plotting_labels,plotting_units)):
            dplot = df_long[df_long['Parameter'] == param]

            # fixed x-axis metabolite order, used both for plotting and for
            # positioning the significance markers
            metab_order = [m for m in cfg['metabolites'] if m in set(dplot['Metabolite'])]

            sns.boxplot(
                data=dplot,
                x='Metabolite',
                y='Value',
                order=metab_order,
                hue='Group',
                hue_order=HUE_ORDER,
                ax=ax,
                #split=False,
                #inner='quart',
                #cut=0
            )
            sns.swarmplot(
                data=dplot,
                x='Metabolite',
                y='Value',
                order=metab_order,
                hue='Group',
                hue_order=HUE_ORDER,
                dodge=True,  # separate points by hue within each metabolite
                alpha=0.6,
                size=3,
                palette="dark:k",
                ax=ax,
                linewidth=0
            )

            # Pairwise group comparisons (Mann-Whitney U + Cohen's d) between the
            # three genotypes, with both raw and BH-FDR-corrected p-values. The FDR
            # correction family is every metabolite x pair test shown on this subplot
            # (i.e. this parameter at this diffusion time).
            pairs = [('WT', 'KI'), ('WT', 'HTZ'), ('KI', 'HTZ')]
            subplot_records = []
            for metab in metab_order:
                for g1, g2 in pairs:
                    a = dplot[(dplot['Metabolite'] == metab) & (dplot['Group'] == g1)]['Value'].dropna().values
                    b = dplot[(dplot['Metabolite'] == metab) & (dplot['Group'] == g2)]['Value'].dropna().values
                    if len(a) >= 2 and len(b) >= 2:
                        try:
                            p_raw = mannwhitneyu(a, b, alternative='two-sided').pvalue
                        except ValueError:  # e.g. all values identical
                            p_raw = np.nan
                        d = cohen_d(a, b)
                    else:
                        p_raw, d = np.nan, np.nan
                    subplot_records.append({
                        'metab': metab, 'pair': f'{g1}-{g2}',
                        'n1': len(a), 'n2': len(b), 'p_raw': p_raw, 'd': d,
                    })

            # FDR-correct across all comparisons on this subplot
            p_fdr = bh_fdr([r['p_raw'] for r in subplot_records])
            for r, pf in zip(subplot_records, p_fdr):
                r['p_fdr'] = pf

            # annotate each metabolite with its significant pairs (|d| >= 0.3 gate);
            # ** = FDR-significant, * = raw-significant only
            for metab in metab_order:
                lines = []
                for r in [r for r in subplot_records if r['metab'] == metab]:
                    if np.isfinite(r['d']) and abs(r['d']) >= 0.3 and np.isfinite(r['p_raw']):
                        if r['p_fdr'] < 0.05:
                            mark = '**'
                        elif r['p_raw'] < 0.05:
                            mark = '*'
                        else:
                            mark = ''
                        r['annotation'] = mark
                        if mark:
                            lines.append(f"{r['pair']}{mark}")
                    else:
                        r.setdefault('annotation', '')
                if lines:
                    ax.annotate("\n".join(lines),
                                xy=(metab_order.index(metab), 0.98),
                                xycoords=('data', 'axes fraction'),
                                ha='center', va='top', color='red', fontsize=8)

            # accumulate for the exported stats table
            for r in subplot_records:
                stats_records.append({
                    'Model': model,
                    'Diffusion Time': dt,
                    'Parameter': param,
                    'Metabolite': r['metab'],
                    'Comparison': r['pair'],
                    'n1': r['n1'],
                    'n2': r['n2'],
                    'p_raw': r['p_raw'],
                    'p_fdr': r['p_fdr'],
                    'cohen_d': r['d'],
                    'annotation': r.get('annotation', ''),
                })

            if i == len(axes) - 1:
                ax.set_xlabel("Metabolite", fontsize=14)
                ax.tick_params(axis='x', rotation=30)
                handles, lgn = ax.get_legend_handles_labels()
                handles, lgn = handles[:len(HUE_ORDER)], lgn[:len(HUE_ORDER)]  # keep only first set (boxplot)
                legend_labels = [f"{l} (n={group_counts[l]})" for l in lgn]
                # the figure-level legend is added below, after the subplot loop
                ax.get_legend().remove()
            else:
                ax.set_xlabel("")
                ax.set_xticklabels([])
                ax.get_legend().remove()

            if model == 'dti':
                ax.set_ylim(0, .2)
            if model == 'stick':
                ax.set_ylim(0, 1.5)
            ax.set_ylabel(
                r"$"+plotting_label+rf"$ [{plotting_unit}]",
                fontsize=14
            )
            ax.tick_params(axis='y', labelsize=14)
            ax.tick_params(axis='x', labelsize=14)

        # single figure-level legend, centred just under the "Metabolite" x-axis
        # label. Placed in figure coordinates (with reserved bottom space via the
        # tight_layout rect below) so its position is consistent across models
        # regardless of how many parameter subplots there are.
        fig.legend(handles, legend_labels, loc='lower center',
                   bbox_to_anchor=(0.5, 0.015), ncol=len(HUE_ORDER), fontsize=12)

        # significance note on its own line, below the legend
        fig.text(0.5, 0.005,
                 "*/** = raw / BH-FDR p<0.05 (|d|>=0.3); pairs WT-KI, WT-HTZ, KI-HTZ",
                 ha='center', fontsize=9)

        if dt == "all":
            fig.suptitle(
                f"{dmrsmodel.model_specs[model].label[:1].capitalize()}{dmrsmodel.model_specs[model].label[1:]}",
                fontsize=20)
            plt.tight_layout(rect=[0, 0.05, 1, 1])  # reserve bottom band for legend + note
            plt.savefig(os.path.join(output_folder, f'{model}_group_comparison.png'),
                        bbox_inches='tight')
        else:
            fig.suptitle(
                rf"{dmrsmodel.model_specs[model].label[:1].capitalize()}{dmrsmodel.model_specs[model].label[1:]} at $\Delta$={int(dt)} ms",
                fontsize=20)
            plt.tight_layout(rect=[0, 0.05, 1, 1])  # reserve bottom band for legend + note
            plt.savefig(os.path.join(output_folder, f'{model}_group_comparison_diffusion_time_{int(dt)}.png'),
                        bbox_inches='tight')
        plt.show()


######## EXPORT GROUP STATISTICS ########
stats_path = output_folder / 'group_stats.csv'
pd.DataFrame(stats_records, columns=[
    'Model', 'Diffusion Time', 'Parameter', 'Metabolite', 'Comparison',
    'n1', 'n2', 'p_raw', 'p_fdr', 'cohen_d', 'annotation',
]).to_csv(stats_path, index=False)
print(f"Saved group statistics to {stats_path}")



