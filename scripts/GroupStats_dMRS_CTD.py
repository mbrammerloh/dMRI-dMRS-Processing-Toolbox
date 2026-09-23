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

# NOTE: must match the subjects actually fitted by main_dMRS_ctd.py (Step2_fitting)
subj_list = [f'sub-{i:02d}' for i in [3,4,5,6,7,8,10,12,14]]    # list of subjects to analyse. Excluded atm: 11 15

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
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx - 1)*np.var(x, ddof=1) + (ny - 1)*np.var(y, ddof=1)) / dof)



######## MODEL-WISE OPERATIONS ########
dmrsmodel = DMRSModel()

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
                            [np.nan] * nb_params,
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

            sns.boxplot(
                data=dplot,
                x='Metabolite',
                y='Value',
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
                hue='Group',
                hue_order=HUE_ORDER,
                dodge=True,  # separate points by hue within each metabolite
                alpha=0.6,
                size=3,
                palette="dark:k",
                ax=ax,
                linewidth=0
            )

            # NOTE: statistical group comparison is not implemented yet. The block
            # below only produces boxplots; the significance-testing code is
            # commented out and still references stale `rois`/`roi` variables
            # (copied from a DWI ROI script) that must be replaced when the
            # actual stats code is added.
            metabs = df_long['Metabolite'].unique()
            for metab in metabs:
                data_roi = df_long[df_long['Metabolite'] == metab]
                group_WT = data_roi[data_roi['Group'] == 'WT']['Value']
                group_KI = data_roi[data_roi['Group'] == 'KI']['Value']

                # Choose test (use Mann-Whitney U if non-normal, or t-test if assumptions met)
                # if not group_WT.empty and not group_KI.empty:
                #     result = mannwhitneyu(group_WT, group_KI, alternative='two-sided')  # non-parametric

                #     pval = result.pvalue
                #     effect_size = cohen_d(group_WT.values, group_KI.values)
                #     if pval < 0.05 and abs(effect_size) >= 0.3:
                #         # Get median y for positioning *
                #         y_max = max(data_roi['Value'].max(), lim[1]) * 1.02
                #         xpos = list(rois).index(roi)
                #         ax.text(x=xpos, y=lim[1]*0.9, s='*', ha='center', va='bottom', fontsize=16, color='red')

            if i == len(axes) - 1:
                ax.set_xlabel("Metabolite", fontsize=14)
                ax.tick_params(axis='x', rotation=30)
                handles, lgn = ax.get_legend_handles_labels()
                handles, lgn = handles[:len(HUE_ORDER)], lgn[:len(HUE_ORDER)]  # keep only first set (boxplot)
                new_lgn = []
                for l in lgn:
                    count = group_counts[l]
                    new_lgn.append(f"{l} (n={count})")
                ax.legend(handles, new_lgn, loc='upper right', fontsize=14)
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

        if dt == "all":
            fig.suptitle(
                f"{dmrsmodel.model_specs[model].label[:1].capitalize()}{dmrsmodel.model_specs[model].label[1:]}",
                fontsize=20)
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, f'{model}_group_comparison.png'))
        else:
            fig.suptitle(
                rf"{dmrsmodel.model_specs[model].label[:1].capitalize()}{dmrsmodel.model_specs[model].label[1:]} at $\Delta$={int(dt)} ms",
                fontsize=20)
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, f'{model}_group_comparison_diffusion_time_{int(dt)}.png'))
        plt.show()



