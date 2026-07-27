import os
import warnings
from typing import Union

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.ticker as mtick
import tqdm.auto as tqdm
from matplotlib.backends.backend_pdf import PdfPages


def alloimmunisation_stats(file_path):
    output_df = pd.read_csv(file_path, sep='\t', index_col=0)
    allo_stat = output_df.loc['allo_avg', :].sum(), np.sqrt(((output_df.loc['allo_stderr', :]).to_numpy()**2).sum())
    return allo_stat

def stats(file_path, stat='allo'):
    output_df = pd.read_csv(file_path, sep='\t', index_col=0)
    out_stat = output_df.loc[f'{stat}_avg', :].sum(), np.sqrt(((output_df.loc[f'{stat}_stderr', :]).to_numpy()**2).sum())
    return out_stat

def objectives_stats(file_path):
    output_df = pd.read_csv(file_path, sep='\t')
    columns = list(output_df.columns)
    return_value = []
    for column in columns:
        return_value.append((output_df[column].mean(), 
                             output_df[column].std() / np.sqrt(len(output_df))))
    return return_value

def objectives_stats_table(files: dict, separate_vals_errs=False, objectives=None):
    data_dict = dict()
    val_dict = dict()
    err_dict = dict()
    indices = ['alloimmunisations', 'scd_shortages', 'expiries', 'all_shortages', 'moves',
               'scd_moves', 'other_moves',
               'O_neg_level', 'O_pos_level', 'O_level',
               'D_subs_num_patients', 'ABO_subs_num_patients', 'ABOD_subs_num_patients']
    if objectives is not None:
        indices = objectives
    for key, value in files.items():
        if value.endswith('_output.tsv'):
            value = value.replace('_output.tsv', '_objectives.tsv')
        value = os.path.realpath(os.path.expanduser(value))
        objstats = objectives_stats(value)
        objstats_str = [f'{x[0]:.3f} ± {x[1]:.3f}' for x in objstats]
        data_dict[key] = objstats_str
        objstats_arr = np.array(objstats)
        val_dict[key] = objstats_arr[:, 0]
        err_dict[key] = objstats_arr[:, 1]
    df = pd.DataFrame(data_dict, index=indices)
    if separate_vals_errs:
        df_val = pd.DataFrame(val_dict, index=indices)
        df_err = pd.DataFrame(err_dict, index=indices)
        return df, (df_val, df_err)
    return df


def read_and_average_comp_times(files: dict):
    data_dict = dict()
    for key, value in files.items():
        if value.endswith('_output.tsv'):
            value = value.replace('_output.tsv', '_computation_times.tsv')
        value = os.path.realpath(os.path.expanduser(value))
        times = np.loadtxt(value, delimiter='\t')
        mean = times.mean()
        std_dev = times.std()
        data_dict[key] = f'{mean:.0f} ± {std_dev:.1f}'
    df = pd.DataFrame(data_dict, index=['computation_time'])
    return df

def objectives_stats_comparison_chart(files: dict, baseline: str, figsize=(12, 8), dpi=200, rows=None, cols=None):
    _, (df_val, df_err) = objectives_stats_table(files, separate_vals_errs=True)

    # Calculate percent change from baseline for df_val and df_err
    df_val_pct_change = df_val.div(df_val[baseline], axis=0) - 1
    # df_err_pct_change = df_err.div(df_err[baseline], axis=0) - 1
    
    # Calculate the error propagation
    df_err_pct_change = (df_val_pct_change + 1) * ((df_err / df_val) ** 2 + (df_err[baseline] / df_val[baseline]) ** 2) ** 0.5

    # Drop the baseline column
    df_val_pct_change =  df_val_pct_change.fillna(0)
    df_err_pct_change = df_err_pct_change.fillna(0)
    df_val_pct_change = df_val_pct_change.drop(baseline, axis=1)
    df_err_pct_change = df_err_pct_change.drop(baseline, axis=1)

    # Transpose the dataframes for plotting
    df_val_pct_change = df_val_pct_change.transpose()
    df_err_pct_change = df_err_pct_change.transpose()
    
    # Create a color palette
    colours = sns.color_palette('deep', df_val_pct_change.shape[1])
    
    # Create a subplot for each objective
    rows = df_val_pct_change.shape[1] if rows is None else rows
    cols = 1 if cols is None else cols
    fig, axs = plt.subplots(rows, cols, figsize=figsize, dpi=dpi)
    for i, (objective, ax) in enumerate(zip(df_val_pct_change.columns, axs.flatten())):
        df_val_pct_change[objective].plot(kind='bar', yerr=df_err_pct_change[objective], ax=ax, color=colours[i])
        ax.set_ylabel('Percent Change')
        # ax.set_xlabel('Scenario')
        ax.grid(axis='y')
        ax.set_title(f'Percent Change in {objective} Relative to Baseline')

        # Set the x-label only for the last subplots
        if i/cols >= rows - 1:
            ax.set_xlabel('Objective')
        else:
            ax.set_xticklabels([])
        
        # Format the y-axis as percentages
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            
    fig.tight_layout()

    return fig, axs


def antigens(rule='Extended', folder='scratch/misc/', courier_costs='data/locations/courier_costs.tsv',):
    from BSCSimulator.experiments.allo_incidence import Antigens, ANTIGENS, DataIO

    bsc_data = DataIO(folder)
    matching_rules = bsc_data.load_rule_sets()
    allo_ab_data = bsc_data.load_alloantibodies()
    immuno = bsc_data.load_immunogenicity()
    mismatch = bsc_data.load_mismatch_weights()
    bsc_data.load_transportation_data(courier_costs)
    
    matching_antigens = matching_rules[rule]['antigen_set']
    compatible_antigens = matching_rules[rule]['compatible']
    mismatch_antigens = matching_rules[rule]['mismatch']
    substitution_antigens = matching_rules[rule]['substitution']
    
    antigens = Antigens(ANTIGENS, rule=matching_antigens, alloimmunisation=immuno,
                        mismatch_weights=mismatch, 
                        allo_Abs=allo_ab_data.values.flatten(),
                        compatible_matching_antigens=compatible_antigens,
                        mismatch_penalisation_antigens=mismatch_antigens,
                        substitution_penalisation_antigens=substitution_antigens,)
    return antigens

def exp2_figs(abod, limited, extended, prob_neg_phen=1, figsize=None, labels=['ABOD', 'Limited', 'Extended'],
              colours=['C0', 'C1', 'C2'], ylabels=None, pdfs=None, skip_plots=False):
    """Figures for Experiment 2

    :param abod:
    :param limited:
    :param extended:
    :param prob_neg_phen:
    :param figsize:
    :return:
    """

    abod_df = pd.read_csv(abod, sep='\t', index_col=0)
    limited_df = pd.read_csv(limited, sep='\t', index_col=0)
    extended_df = pd.read_csv(extended, sep='\t', index_col=0)

    labels_mismatch = ["A", "B", "D", "C", "c", "E", "e", "K", "k", "Fya", "Fyb", "Jka", "Jkb", "M", "N", "S", "s"]
    labels_allo = labels_mismatch[3:]
    x1 = np.arange(len(labels_mismatch))
    x2 = np.arange(len(labels_allo))
    width = 0.2

    x = [x2, x2, x2]
    met = ['mismatch', 'subs', 'allo']
    ylabel = ['Number of mismatches', 'Expected number of substitutions', 'Expected number of alloimmunisations']
    ylabel = ylabels if ylabels is not None else ylabel
    title = ['Mismatches in antigen-negative patients by antigen and matching rule',
             'Substitutions per unit transfused by antigen and matching rule',
             'Alloimmunisations by antigen and matching rule']
    # xticks = [labels_mismatch, labels_mismatch, labels_allo]
    xticks = [labels_allo, labels_allo, labels_allo]
    cols = xticks

    for i in range(len(x)):
        if skip_plots is True or skip_plots[i]:
            continue
        if met[i] == 'subs':
            _neg_phen = 1
        else:
            _neg_phen = prob_neg_phen
        # fig, ax = plt.subplots(figsize=(12, 8))
        fig, ax = plt.subplots(figsize=figsize, dpi=200)
        ax.bar(x[i] - width, abod_df.loc[met[i] + '_avg', cols[i]].to_numpy() / _neg_phen, width, label=labels[0],
               yerr=abod_df.loc[met[i] + '_stderr', cols[i]].to_numpy() / _neg_phen, color=colours[0])
        ax.bar(x[i], limited_df.loc[met[i] + '_avg', cols[i]].to_numpy() / _neg_phen, width, label=labels[1],
               yerr=limited_df.loc[met[i] + '_stderr', cols[i]].to_numpy() / _neg_phen, color=colours[1])
        ax.bar(x[i] + width, extended_df.loc[met[i] + '_avg', cols[i]].to_numpy() / _neg_phen, width, label=labels[2],
               yerr=extended_df.loc[met[i] + '_stderr', cols[i]].to_numpy() / _neg_phen, color=colours[2])
        # ax.set_ylabel(ylabel[i], fontsize=14)
        ax.set_ylabel(ylabel[i])
        # ax.set_title(title[i], fontsize=18)
        #ax.set_title(title[i])
        ax.set_xticks(x[i])
        # ax.set_xticklabels(xticks[i], fontsize=14)
        ax.set_xticklabels(xticks[i])
        if met[i] == 'subs':
            ax.legend()
        else:
            ax.legend()
            # ax.legend(bbox_to_anchor=(1.0, 1.0))
        fig.tight_layout()
        plt.show()
        if pdfs is not None and pdfs[i] is not None:
            fig.savefig(pdfs[i], bbox_inches='tight')
            plt.close(fig)


def exp3_figs(limited, extended, prob_neg_phen=1, figsize=(12, 8), labels=['Limited', 'Extended']):

    limited_df = pd.read_csv(limited, sep='\t', index_col=0)
    extended_df = pd.read_csv(extended, sep='\t', index_col=0)

    labels_mismatch = ["A", "B", "D", "C", "c", "E", "e", "K", "k", "Fya", "Fyb", "Jka", "Jkb", "M", "N", "S", "s"]
    labels_allo = labels_mismatch[3:]
    x1 = np.arange(len(labels_mismatch))
    x2 = np.arange(len(labels_allo))
    width = 0.35

    x = [x2, x2, x2]
    met = ['mismatch', 'subs', 'allo']
    ylabel = ['Number of mismatches', 'Expected number of substitutions', 'Expected number of alloimmunisations']
    title = ['Mismatches in antigen-negative patients by antigen and matching rule', 'Substitutions per unit transfused by antigen and matching rule',
             'Alloimmunisations by antigen and matching rule']
    # xticks = [labels_mismatch, labels_mismatch, labels_allo]
    xticks = [labels_allo, labels_allo, labels_allo]
    cols = xticks

    for i in range(len(x)):
        if met[i] == 'subs':
            _neg_phen = 1
        else:
            _neg_phen = prob_neg_phen
        fig, ax = plt.subplots(figsize=figsize, dpi=200)
        ax.bar(x[i] - width/2, limited_df.loc[met[i] + '_avg', cols[i]].to_numpy() / _neg_phen, width, label=labels[0],
               yerr=limited_df.loc[met[i] + '_stderr', cols[i]].to_numpy() / _neg_phen, color='tab:blue')
        ax.bar(x[i] + width/2, extended_df.loc[met[i] + '_avg', cols[i]].to_numpy() / _neg_phen, width, label=labels[1],
               yerr=extended_df.loc[met[i] + '_stderr', cols[i]].to_numpy() / _neg_phen, color='tab:green')
        # ax.set_ylabel(ylabel[i], fontsize=14)
        ax.set_ylabel(ylabel[i])
        # ax.set_title(title[i], fontsize=18)
        # ax.set_title(title[i])
        ax.set_xticks(x[i])
        # ax.set_xticklabels(xticks[i], fontsize=14)
        ax.set_xticklabels(xticks[i])
        ax.legend()
        fig.tight_layout()
        plt.show()
        
def create_histogram_panels(data1, data2, data3):
    # Create a figure with three subplots
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 12), sharex=True)

    # Plot histogram for dataset 1
    axes[0].hist(data1, bins=20, color='blue', alpha=0.5)
    axes[0].set_title('Limited Rule')

    # Plot histogram for dataset 2
    axes[1].hist(data2, bins=20, color='green', alpha=0.5)
    axes[1].set_title('Extended Rule')

    # Plot histogram for dataset 3
    axes[2].hist(data3, bins=20, color='red', alpha=0.5)
    axes[2].set_title('Optimised Extended Rule')

    # Label axes
    for ax in axes:
        ax.set_ylabel('Frequency')
    axes[2].set_xlabel('Total expected alloimmunisations')

    # Adjust spacing between subplots
    fig.tight_layout()

    # Display the chart
    plt.show()
    
def create_histograms(data1, data2, data3, xlabel='Total expected alloimmunisations', title='Distribution of total expected alloimmunisations',
                      labels=['Limited Rule', 'Extended Rule', 'Extended Anticipation Rule'], density=False, xlim=None):
    # Create a figure and axes for the panel
    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)

    # Using seaborn.histplot instead
    sns.histplot(data1, color='tab:blue', alpha=0.5, label=labels[0], ax=ax, kde=density)
    sns.histplot(data2, color='tab:green', alpha=0.5, label=labels[1], ax=ax, kde=density)
    sns.histplot(data3, color='tab:red', alpha=0.5, label=labels[2], ax=ax, kde=density)

    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    
    if xlim is not None:
        ax.set_xlim(*xlim)
    # ax.set_xlim(0, 12)

    # Add legend
    ax.legend()

    fig.tight_layout()
    
    # Display the chart
    plt.show()
    

def get_data_from_file_for_histograms(filenames: list, data_type: str):
    """
    Get data from files for histograms
    :param filenames: list of filenames
    :param data_type: column name in the file
    :return: tuple of arrays of data
    """
    data = []
    for filename in filenames:
        if filename.endswith('_output.tsv'):
            filename = filename.replace('_output.tsv', '_objectives.tsv')
        df = pd.read_csv(os.path.realpath(os.path.expanduser(filename)), sep='\t')
        data.append(df[data_type].to_numpy())
    return tuple(data)


def create_stock_levels_graph(datafilename, columns, labels=None, xlabel='Time (days)',
                              ylabel='Number of units', figsize=(8, 6), dpi=300,
                              ylim=None, xlim=None, raw_data=False):
    """
    Create a graph of stock levels
    :param datafilename: name of the file with data
    :param columns: list of columns to plot
    :param labels: list of labels for columns
    :param xlabel: label for x axis
    :param ylabel: label for y axis
    :param figsize: size of the figure
    :param dpi: resolution of the figure
    :param ylim: tuple of (lower, upper) bounds for the y-axis
    :return: None
    """
    if datafilename.endswith('_output.tsv'):
        datafilename = datafilename.replace('_output.tsv', '_stocks.tsv')
    df = pd.read_csv(os.path.realpath(os.path.expanduser(datafilename)), sep='\t')
    if raw_data:
        df = df[columns] * df['total'].values[:, None]
    else:
        df = df[columns]
    if labels is not None:
        df.columns = labels
    fig = plt.figure(dpi=dpi)
    ax = fig.gca()
    # ax.grid(True)
    ax = df.plot(figsize=figsize, kind='line', xlabel=xlabel, ylabel=ylabel, ax=ax, grid=True)
    if not raw_data:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)
    

def stacked_stock_levels_graph(datafilename, columns, labels=None, bbox_to_anchor=(1.01, 0.5),
                               xlabel='Time (days)', ylabel='Level of total stock', figsize=(8, 6), dpi=200,
                               demarcate_warmup=False, warmup_color='black', raw_data=False,
                               warmup_period=7*6*3, title=None):
    """
    Create a stacked graph of stock levels
    :param datafilename: name of the file with data
    :param columns: list of columns to plot
    :param labels: list of labels for columns
    :param xlabel: label for x axis
    :param ylabel: label for y axis
    :param figsize: size of the figure
    :param dpi: resolution of the figure
    :return: None
    """
    if datafilename.endswith('_output.tsv'):
        datafilename = datafilename.replace('_output.tsv', '_stocks.tsv')
    df = pd.read_csv(os.path.realpath(os.path.expanduser(datafilename)), sep='\t')
    if raw_data:
        df = df[columns] * df['total'].values[:, None]
    else:
        df = df[columns]
    if labels is not None:
        df.columns = labels
    fig, ax = plt.subplots(dpi=dpi, figsize=figsize)
    ax.stackplot(np.array(df.index) +1, df.T.to_numpy(), labels=df.columns, alpha=0.75)
    ax.legend(loc='center left', bbox_to_anchor=bbox_to_anchor)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if title is not None or title != '':
        ax.set_title(title)
    if demarcate_warmup:
        draw_warmup_line(ax,color=warmup_color, x=warmup_period+0.5)
    if not raw_data:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    fig.tight_layout()
    

def stacked_stock_levels_locations_graph(datafilename, phenotypes, locations, subplots,
                                         phen_labels=None, loc_labels=None, 
                                         bbox_to_anchor=(1.01, 0.5), title='',
                                         xlabel='Time (days)', ylabel='Level of total stock',
                                         figsize=(8, 6), dpi=200, demarcate_warmup=False,
                                         warmup_color='black', raw_data=False, warmup_period=7*6*3,
                                         days_range=np.arange(1, 211), plot_bgrps=False, y_lim=None,
                                         avail_stock=False,):
    if datafilename.endswith('_output.tsv'):
        datafilename = datafilename.replace('_output.tsv', '_stocks_locs.npz')
    data = np.load(os.path.realpath(os.path.expanduser(datafilename)))
    phenotypes_to_plot = [True if p in phenotypes else False for p in data['watched_phenotypes_names']]
    locations_to_plot = [True if l in locations else False for l in data['locations']]
    phen_labels = phen_labels if phen_labels is not None else data['watched_phenotypes_names'][phenotypes_to_plot]
    # print(len(locations_to_plot)); print(len(phenotypes_to_plot))
    loc_labels = loc_labels if loc_labels is not None else data['locations'][locations_to_plot]
    data_key = 'stock_loc' if not avail_stock else 'avail_stock_loc'
    # locs_to_plot = np.arange(data[f'{data_key}_means'].shape[1])[locations_to_plot]
    # phens_to_plot = np.arange(data[f'{data_key}_means'].shape[2])[phenotypes_to_plot]
    data_means = data[f'{data_key}_means'][:, locations_to_plot, :][:, :, phenotypes_to_plot]
    # data_stds = data[f'{data_key}_stderrs'][:, locations_to_plot, :][:, :, phenotypes_to_plot]
    if not raw_data:
        if not plot_bgrps:
            data_means = data_means / data_means.sum(axis=2, keepdims=True)
        else:
            data_means = data_means / data_means.sum(axis=1, keepdims=True)        
    
    fig, axs = plt.subplots(subplots[0], subplots[1], figsize=figsize, dpi=dpi, sharex='col')
    if not plot_bgrps:
        for i, (ax, _) in enumerate(zip(axs.flatten(), locations)):
            ax.stackplot(days_range, data_means[:,i,:].T, labels=phen_labels, alpha=0.75)
            # ax.legend(loc='center left', bbox_to_anchor=bbox_to_anchor)
            # ax.set_ylabel(ylabel)
            # ax.set_xlabel(xlabel)
            ax.set_title(loc_labels[i])
            if demarcate_warmup:
                draw_warmup_line(ax,color=warmup_color, x=warmup_period+0.5)
            if not raw_data:
                ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            ax.set_ylim(0, y_lim)
        fig.supxlabel(xlabel)
        fig.supylabel(ylabel)
        fig.suptitle(title)
        fig.legend(phen_labels, loc='center left', bbox_to_anchor=bbox_to_anchor, bbox_transform=fig.transFigure)
        # fig.tight_layout()
    else:
        for i, (ax, _) in enumerate(zip(axs.flatten(), phenotypes)):
            ax.stackplot(days_range, data_means[:,:,i].T, labels=loc_labels, alpha=0.75)
            # ax.legend(loc='center left', bbox_to_anchor=bbox_to_anchor)
            # ax.set_ylabel(ylabel)
            # ax.set_xlabel(xlabel)
            ax.set_title(phen_labels[i])
            if demarcate_warmup:
                draw_warmup_line(ax,color=warmup_color, x=warmup_period+0.5)
            if not raw_data:
                ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        fig.supxlabel(xlabel)
        fig.supylabel(ylabel)
        fig.suptitle(title)
        fig.legend(loc_labels, loc='center left', bbox_to_anchor=bbox_to_anchor, bbox_transform=fig.transFigure)
    

def plot_age_dist_blood(datafilename, array='age_dist_given_to_scd', age_range=None,
                                 days_range=None, title='Age distribution of blood given to SCD patients over time',
                                 vmax=300, cmap='viridis', figsize=(15, 6), dpi=200, re_index=False,
                                 print_max=False, every_n_xtick=2, plot_avg=False, print_mean=False,
                                 pdf=None):
    if datafilename.endswith('_output.tsv'):
        datafilename = datafilename.replace('_output.tsv', '_age_distributions.npz')
    data = np.load(os.path.realpath(os.path.expanduser(datafilename)))
    age_dist = data[array]
    # Mask the day 0 and day 1 data instead of clipping it out
    warnings.warn(
        "⚠️  WARNING: This function does not automatically assume ZERO-INDEXED ages. "
        "If you are working with newer data using 0-indexed ages, results may be incorrect. "
        "Ensure your data format matches the expected indexing convention.",
        UserWarning,
        stacklevel=2
    )
    age_range = age_range if age_range is not None else np.arange(1, 15)
    days_range = days_range if days_range is not None else np.arange(1, 211)
    days_labels = days_range if not re_index else np.arange(len(days_range)) + 1
    arr = age_dist[days_range-1,:]
    arr = arr[:,age_range]
    sba_df = pd.DataFrame(arr.T, index=age_range, columns=days_labels)
    if plot_avg:
        plot_avg_age_dist_blood_scd(sba_df, title=title, ylim_max=vmax,
                                    figsize=figsize, dpi=dpi, pdf=pdf, print_mean=print_mean)
        return sba_df
    # Mask the zero values instead of making them NaNs
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    sns.heatmap(sba_df.replace(0, np.nan), cmap=cmap, ax=ax, cbar_kws={'label': 'Number of units'}, vmax=vmax)
    ax.set_ylabel('Age (days)')
    ax.set_xlabel('Time (days)')
    ax.invert_yaxis()
    ax.set_title(title)
    ax.grid()
    # Only show every other xtick label
    for i, label in enumerate(ax.xaxis.get_ticklabels()):
        if i % every_n_xtick != 0:
            label.set_visible(False)
    fig.tight_layout()
    fig.show()
    data.close()
    if print_max:
        print(sba_df.max(None, skipna=True))
    if pdf is not None:
        fig.savefig(pdf, bbox_inches='tight')
        
def plot_avg_age_dist_blood_scd(df, title='Mean age distribution of blood given to SCD patients',
                                ylim_max=None, figsize=(8, 6), dpi=200, pdf=None, print_mean=False):
    warnings.warn(
    "⚠️  WARNING: This function does not automatically assume ZERO-INDEXED ages. "
    "If you are working with newer data using 0-indexed ages, results may be incorrect. "
    "Ensure your data format matches the expected indexing convention.",
    UserWarning,
    stacklevel=2
    )
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    df_hist = df.mean(axis=1)
    df_hist_std_err = df.std(axis=1) / np.sqrt(df.count(axis=1))
    i_min = df_hist.index.min()
    i_max = df_hist.index.max()
    ax.stairs(df_hist.to_numpy().flatten(), np.arange(i_min, i_max+2) - 0.5, fill=True, color='gray', alpha=1)
    # ax.errorbar(df_hist.index, df_hist, yerr=df_hist_std_err, fmt='o', color='black')
    ax.stairs(df_hist.to_numpy().flatten() + df_hist_std_err.to_numpy().flatten(), 
                np.arange(i_min, i_max+2) - 0.5, fill=False, color='black', alpha=0.35, ls=':')
    ax.stairs(df_hist.to_numpy().flatten() - df_hist_std_err.to_numpy().flatten(),
                np.arange(i_min, i_max+2) - 0.5, fill=False, color='black', alpha=0.35, ls=':')
    print(df_hist_std_err.to_numpy().flatten().max())
    # print(df_hist_std_err)
    ax.grid()
    ax.set_xlabel('Age (days)')
    ax.set_ylabel('Number of units')
    ax.set_xlim(0, None)
    ax.set_ylim(0, ylim_max)
    ax.set_title(title)
    fig.tight_layout()
    if pdf is not None:
        fig.savefig(pdf, bbox_inches='tight')
    # fig.show()
    if print_mean:
        print(np.average(df_hist.index, weights=df_hist))
    

def ccdf_avg_age_dist_blood_scd(datafilename, array='age_dist_given_to_scd', age_range=None,
                                days_range=None, above_age=0, re_index=False):
    if datafilename.endswith('_output.tsv'):
        datafilename = datafilename.replace('_output.tsv', '_age_distributions.npz')
    warnings.warn(
        "⚠️  WARNING: This function does not automatically assume ZERO-INDEXED ages. "
        "If you are working with newer data using 0-indexed ages, results may be incorrect. "
        "Ensure your data format matches the expected indexing convention.",
        UserWarning,
        stacklevel=2
    )
    data = np.load(os.path.realpath(os.path.expanduser(datafilename)))
    age_dist = data[array]
    age_range = age_range if age_range is not None else np.arange(1, 15)
    days_range = days_range if days_range is not None else np.arange(1, 211)
    days_labels = days_range if not re_index else np.arange(len(days_range)) + 1
    arr = age_dist[days_range-1,:]
    arr = arr[:,age_range]
    sba_df = pd.DataFrame(arr.T, index=age_range, columns=days_labels)
    df_hist = sba_df.mean(axis=1)
    i_min = df_hist.index.min()
    i_max = df_hist.index.max()
    tot_mass = df_hist.loc[i_min:i_max].sum()
    mass_above = df_hist[df_hist.index > above_age].sum()
    # df_hist = df_hist.sort_index(ascending=False)
    # ccdf = df_hist.cumsum()
    # ccdf = ccdf / ccdf.max()
    ccdf = mass_above / tot_mass
    return ccdf


def get_age_dist(datafilename, array,):
    if datafilename.endswith('_output.tsv'):
        datafilename = datafilename.replace('_output.tsv', '_age_distributions.npz')
    warnings.warn(
        "⚠️  WARNING: This function does not automatically assume ZERO-INDEXED ages. "
        "If you are working with newer data using 0-indexed ages, results may be incorrect. "
        "Ensure your data format matches the expected indexing convention.",
        UserWarning,
        stacklevel=2
    )
    data = np.load(os.path.realpath(os.path.expanduser(datafilename)))    
    age_dist = data[array]
    data.close()
    return age_dist
    

def draw_warmup_line(ax, x=168.5, color='black', linestyle='--', linewidth=2, text="Warm-up Period",
                     xy=None, xytext=None):
    y = np.mean(ax.get_ylim())
    if xy is None:
        xy = (x, y)
    if xytext is None:
        xytext = (x *0.99, y)
    ax.axvline(x=x, color=color, linestyle=linestyle, linewidth=linewidth)
    ax.annotate(text, xy=xy, xytext=xytext, rotation='vertical',
                color=color, ha='right', va='center', textcoords='data')
    return ax


def reshape_abo_combos(datafilename, mixed_match=False, stderr=0):
    if mixed_match:
        suffix = '_abodmm_subs.tsv'
    else:
        suffix = '_abocm.tsv'
    if datafilename.endswith('_output.tsv'):
        datafilename = datafilename.replace('_output.tsv', suffix)
    labels = ['O-', 'O+', 'B-', 'B+', 'A-', 'A+', 'AB-', 'AB+']
    data = pd.read_csv(os.path.realpath(os.path.expanduser(datafilename)), sep='\t', nrows=2, index_col=0)
    return pd.DataFrame(data.values[stderr,:].reshape(8,8), columns=labels, index=labels)    

def add_totals_to_abo_combos(reshaped_abo_combos):
    reshaped_abo_combos['Total'] = reshaped_abo_combos.sum(axis=1)
    reshaped_abo_combos.loc['Total'] = reshaped_abo_combos.sum(axis=0)
    return reshaped_abo_combos


def mixed_allocation_totals(datafilename, print_with_stderr=False, breakdowns=False):
    # reshaped_means = reshape_abo_combos(datafilename, mixed_match=True)
    if datafilename.endswith('_output.tsv'):
        datafilename = datafilename.replace('_output.tsv', '_abodmm_subs.tsv')
    data = pd.read_csv(os.path.realpath(
        os.path.expanduser(datafilename)), sep='\t', nrows=2, index_col=0)
    abo_x = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    abo_mm_indices = abo_x > abo_x[:, None]
    d_x = np.array([0, 1] * 4)
    d_mm_indices = d_x > d_x[:, None]
    abod_mm_indices = abo_mm_indices & d_mm_indices
    indices = (d_mm_indices, abo_mm_indices, abod_mm_indices)
    compatible_indices = (
                            [True, True, True, True,                # O-
                             False, True, False, True,              # B-
                             False, False, True, True,              # A-
                             False, False, False, True],            # AB-
                            [True, True, True, True, True, True,    # O-
                            False, True, False, True, False, True,  # O+
                            False, False, True, True,               # B-
                            False, False, False, True,              # B+
                            True, True,                             # A-
                            False, True],                           # A+
                            [True, True, True,                      # O-
                             False, True,                           # B-
                             True]                                  # A-
                          )
    if breakdowns:
        return_values = []
        for i, ci in zip(indices, compatible_indices):
            return_values.append(data.loc[:, i.flatten()].loc[:, ci])
        return tuple(return_values)
    
    return_values = []
    for i in indices:
        values = data.loc[:, i.flatten()].sum(axis=1)
        value = values[0]
        if print_with_stderr:
            value = f'{values[0]:.1f} ± {values[1]:.2f}'
        return_values.append(value)
    return tuple(return_values)
    

def abo_combos_usage_demand(totalled_combos):
    demanded = pd.DataFrame((totalled_combos.values[-1, :-1]/totalled_combos.values[-1, :-1].sum())[None, :], columns=totalled_combos.columns[:-1], index=['Demand'])
    used = pd.DataFrame((totalled_combos.values[:-1, -1]/totalled_combos.values[:-1, -1].sum())[None, :], columns=totalled_combos.columns[:-1], index=['Usage'])
    return pd.concat([demanded, used], axis=0)
    

def compile_tuning_points_all_objectives(folder, pattern='_objectives.tsv',
                                         suffix='tuning_all-vars-objs.tsv'):
    files = os.listdir(folder)
    tuning_points = [f for f in files if f.endswith('tuning_points.tsv')][0]
    files = [os.path.join(folder, x) for x in files if x.endswith(pattern)]
    files.sort(key=lambda x: os.path.getctime(os.path.join(folder, x)))

    averages = []
    for file in files:
        # Load the file into a pandas dataframe
        df = pd.read_csv(os.path.join(folder, file), sep='\t')
        
        # Calculate the column averages
        column_averages = df.mean()
        
        # Add the column averages to the list of averages
        averages.append(column_averages)

    # Concatenate the list of averages into a single dataframe
    averages_df = pd.concat(averages, axis=1)
    averages_df = averages_df.transpose()
    all_vars = pd.read_csv(os.path.join(folder,
                                        tuning_points),
                            sep='\t')
    var_names = ['immunogenicity', 'usability', 'substitutions', 'fifo', 'young_blood']
    all_vars = all_vars[var_names]
    vars_objs = pd.concat([all_vars, averages_df], axis=1)
    vars_objs.to_csv(os.path.join(folder, 
                                  tuning_points.replace('tuning_points.tsv', 
                                                        suffix)),
                        sep='\t', index=False)
    return vars_objs    


def summarise_abo_mixed_allocations(abo_mixed_allocations: pd.DataFrame) -> pd.DataFrame:
    donor_target = [('O', 'B'),
                    ('O', 'A'),
                    ('O', 'AB'),
                    ('B', 'AB'),
                    ('A', 'AB')]
    data = dict()
    for combo in donor_target:
        cols = [col for col in abo_mixed_allocations.columns if (
            combo[0] + '+ ' in col or combo[0] + '- ' in col) and 'to ' + combo[1] in col]
        mean_values = [abo_mixed_allocations[col][0] for col in cols]
        std_err_values = [abo_mixed_allocations[col][1] for col in cols]
        col_vals = [sum(mean_values), np.sqrt(np.square(std_err_values).sum())]
        data.update({f'{combo[0]} to {combo[1]}': col_vals})
    return pd.DataFrame(data, index=['Mean', 'Std. Err.'])


def usability_difference_matrix():
    from BSCSimulator.util import dummy_population_phenotypes, abd_usability
    
    non_scd_frequencies = dummy_population_phenotypes(
        'data/bloodgroup_frequencies/ABD_old_dummy_demand.tsv')
    usability = abd_usability(
        non_scd_frequencies.frequencies.to_numpy(),
        330/3500, 1.0)
    usability_diff = usability[:, None] - usability
    compatibility = [[True] * 8,                       # O-
                     [False, True] * 4,                # O+
                     [False, False, True, True] * 2,   # B-
                     [False, False, False, True] * 2,  # B+
                     [False] * 4 + [True] * 4,         # A-
                     [False] * 4 + [False, True] * 2,  # A+
                     [False] * 6 + [True] * 2,         # AB-
                     [False] * 6 + [False, True]       # AB+
                    ]
    compatibility = np.array(compatibility)
    usability_diff[~compatibility] = np.nan
    blood_groups = ['O-', 'O+', 'B-', 'B+', 'A-', 'A+', 'AB-', 'AB+']
    usability_diff_matrix = pd.DataFrame(usability_diff,
                                         columns=[f'to {bg}' for bg in blood_groups],
                                         index=blood_groups)
    return usability_diff_matrix


def avg_stock_composition(rules, rule_files, donations=None,
                          figsize=(12,5), dpi=200, ncol=4, bbox_to_anchor=(0.5, -0.12),
                          pdf=None):
    import matplotlib.ticker as mtick

    # rules = ['Donations', 'E0', 'E0 (No SUB)']
    # rule_files = [naive_weights, extended_no_substitution]
    blood_grps = ['O-', 'O+', 'B-', 'B+', 'A-', 'A+', 'AB-', 'AB+']
    if donations is None:
        donations = np.array( [14.6, 36.2, 2.8, 7.8, 7.8, 28.4, 0.6, 1.8])
    data = [donations / donations.sum()]
    files = []
    for rule in rule_files:
        stock = pd.read_csv(rule.replace('_output.tsv', '_stocks.tsv'), sep='\t')
        stock = stock[blood_grps].to_numpy()[-42:].mean(axis=0)
        data.append(stock / stock.sum())
    data = np.transpose(data)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    for i in range(data.shape[0]):
        ax.bar(rules, data[i], bottom=np.sum(data[:i], axis=0), label=blood_grps[i])
    ax.set_ylabel('Proportion of Total Stock')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    # ax.set_title('Average Blood Stock Composition')
    ax.legend(ncol=ncol, bbox_to_anchor=bbox_to_anchor, loc='upper center')
    # ax.grid()
    fig.tight_layout()
    if pdf is not None:
        fig.savefig(pdf, bbox_inches='tight')
    plt.show()


def compile_moves_matrix(files: Union[list, str], shus: dict, warm_up = 0, 
                         dest_col=9, move_reasons:dict={}, date_moved_col=6, pat_grp_col=8, origin_col=3,):
    if isinstance(files, str):
        files = [files]
    ids = shus.keys()
    names = shus.values()
    all_shu_combinations = list_of_permutations([tuple(ids), tuple(ids)])
    ids_to_index = {id: i for i, id in enumerate(ids)}
    for f in tqdm.tqdm(files, desc='Files', position=0):
        if f.endswith('_output.tsv'):
            f = f.replace('_output.tsv', '_moves.npz')
        compiled_file_mean = f.replace('_moves.npz', '_compiled_moves.tsv')
        compiled_file_stderr = f.replace('_moves.npz', '_compiled_moves_stderr.tsv')
        others_compiled_files_mean = {reason: f.replace('_moves.npz', f'_compiled_moves_{reason}.tsv') for reason in move_reasons.keys()}
        others_compiled_files_stderr = {reason: f.replace('_moves.npz', f'_compiled_moves_{reason}_stderr.tsv') for reason in move_reasons.keys()}
        data = np.load(f)
        # others_data_matrices = {reason: np.zeros((len(ids), len(ids))) for reason in move_reasons.keys()}
        # all_data_matrix = np.zeros((len(ids), len(ids)))
        all_data_tensor = []
        others_data_tensors = {reason: [] for reason in move_reasons.keys()}
        
        for replication in tqdm.tqdm(data.files, desc='Replications',position=1, leave=False ):
            all_data_table = np.zeros((len(ids), len(ids)))
            moves = data[replication]
            steady_state_moves = moves[moves[:, date_moved_col] > warm_up]
            for i, j in all_shu_combinations:
                comparison = np.array([i, j])
                truth = np.all(steady_state_moves[:, [origin_col, dest_col]] == comparison, axis=1)
                # all_data_matrix[ids_to_index[i], ids_to_index[j]] += np.sum(truth)
                all_data_table[ids_to_index[i], ids_to_index[j]] += np.sum(truth)
            all_data_tensor.append(all_data_table)
            if len(move_reasons) == 0:
                continue
            for reason, r_id in move_reasons.items():
                others_data_table = np.zeros((len(ids), len(ids)))
                for i, j in all_shu_combinations:
                    comparison = np.array([i, r_id, j])
                    truth = steady_state_moves[:, [origin_col, pat_grp_col, dest_col]] == comparison
                    truth = np.all(truth, axis=1)
                    # others_data_matrices[reason][ids_to_index[i], ids_to_index[j]] += np.sum(truth)
                    others_data_table[ids_to_index[i], ids_to_index[j]] += np.sum(truth)
                others_data_tensors[reason].append(others_data_table)
        
        # data_matrix = pd.DataFrame(all_data_matrix / len(data.files), index=names, columns=names)
        # data_matrix.to_csv(compiled_file_mean, sep='\t')
        # print(f'Compiled moves matrix for {f} saved to {compiled_file_mean}')
        data_mean_matrix = pd.DataFrame(np.mean(all_data_tensor, axis=0), index=names, columns=names)
        data_mean_matrix.to_csv(compiled_file_mean, sep='\t')
        # print(f'Compiled mean moves matrix for {f} saved to {compiled_file_mean}')
        data_stderr_matrix = pd.DataFrame(np.std(all_data_tensor, axis=0) / np.sqrt(len(all_data_tensor)),
                                            index=names, columns=names)
        data_stderr_matrix.to_csv(compiled_file_stderr, sep='\t')
        # print(f'Compiled standard error moves matrix for {f} saved to {compiled_file_stderr}')

        for reason, matrix in others_data_tensors.items():
            # matrix = matrix / len(data.files)
            # matrix_df = pd.DataFrame(matrix, index=names, columns=names)
            # matrix_df.to_csv(others_compiled_files_mean[reason], sep='\t')
            # print(f'Compiled moves matrix for {reason} in {f} saved to {others_compiled_files_mean[reason]}')
            # reason_mean_matrix = others_data_tensors[reason]
            mean_matrix_df = pd.DataFrame(np.mean(matrix, axis=0), 
                                                index=names, columns=names)
            mean_matrix_df.to_csv(others_compiled_files_mean[reason], sep='\t')
            # print(f'Compiled mean moves matrix for {reason} in {f} saved to {others_compiled_files_mean[reason]}')
            stderr_matrix_df = pd.DataFrame(np.std(matrix, axis=0) / np.sqrt(len(matrix)),
                                                index=names, columns=names)
            stderr_matrix_df.to_csv(others_compiled_files_stderr[reason], sep='\t')
            # print(f'Compiled standard error moves matrix for {reason} in {f} saved to {others_compiled_files_stderr[reason]}')
        data.close()


def plot_moves_matrix(files: Union[list, str], figsize=(10,8), annotate=True,
                      cmap="YlGnBu", pdfs=None, dpi=200, titles=None,
                      fmt='.0f', moves_reason='', order_of_shus=None, trim_shu_label_name=True,
                      vmax=None, return_matrices=False, data_divisor=1, fontsize=None,):
    if isinstance(files, str):
        files = [files]
        titles = [titles]
        pdfs = [pdfs]
    if len(titles) < len(files):
        titles += [None] * (len(files) - len(titles))
    if len(pdfs) < len(files):
        pdfs += [None] * (len(files) - len(pdfs))
    matrices = []
    for f, title, pdf in zip(files, titles, pdfs):
        if f.endswith('_output.tsv') and moves_reason == '':
            f = f.replace('_output.tsv', '_compiled_moves.tsv')
        elif f.endswith('_output.tsv') and moves_reason != '':
            f = f.replace('_output.tsv', f'_compiled_moves_{moves_reason}.tsv')
        data = pd.read_csv(f, sep='\t', index_col=0)
        if order_of_shus is not None:
            data = data.iloc[order_of_shus, order_of_shus]
        if trim_shu_label_name:
            rename_dict = {name: name[name.find('_')+1:] for name in data.columns}
            data = data.rename(columns=rename_dict, index=rename_dict)
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        sns.heatmap(data / data_divisor, ax=ax, cmap=cmap, annot=annotate, fmt=fmt, vmax=vmax,)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=fontsize)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right', fontsize=fontsize)
        ax.set_title(title)
        ax.set_xlabel('Destination SHU', fontsize=fontsize)
        ax.set_ylabel('Source SHU', fontsize=fontsize)
        if pdf is not None:
            fig.savefig(pdf, bbox_inches='tight')
        fig.show()
        matrices.append(data)
    if return_matrices:
        return matrices
    # fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    # sns.heatmap(pd.read_csv(files[0], sep='\t', index_col=0), ax=ax, cmap=cmap, annot=annotate)


def calculate_courier_costs(files: Union[list[str], str], cost_matrix: pd.DataFrame,
                            moves_reason='', order_of_shus=None):
    all_courier_costs = {}
    if isinstance(files, str):
        files = [files]
    for f in files:
        if f.endswith('_output.tsv') and moves_reason == '':
            f = f.replace('_output.tsv', '_compiled_moves.tsv')
        elif f.endswith('_output.tsv') and moves_reason != '':
            f = f.replace('_output.tsv', f'_compiled_moves_{moves_reason}.tsv')
        data = pd.read_csv(f, sep='\t', index_col=0)
        courier_costs = data * cost_matrix
        if order_of_shus is not None:
            courier_costs = courier_costs.iloc[order_of_shus, order_of_shus]
        all_courier_costs.update({f: courier_costs})
        # all_courier_costs.append(courier_costs)
    return all_courier_costs


def max_number_of_moves_in_a_day(files: Union[list[str], str], shus: dict, warm_up = 0, move_reason=3,
                                origin_col=3, date_moved_col=6, pat_grp_col=8, ):
    if isinstance(files, str):
        files = [files]
    ids = shus.keys()
    names = list(shus.values())
    results = {}
    for f in files:
        _f = f
        if f.endswith('_output.tsv'):
            f = f.replace('_output.tsv', '_moves.npz')
        data = np.load(f)
        max_moves_shus = []
        max_moves = []
        for replication in data.files:
            shus_max_moves = np.zeros(len(ids))
            moves = data[replication]
            steady_state_moves = moves[moves[:, date_moved_col] > warm_up]
            if type(move_reason) == int:
                steady_state_moves = steady_state_moves[steady_state_moves[:, pat_grp_col] == move_reason]
            for i, id in enumerate(ids):
                shu_moves = steady_state_moves[steady_state_moves[:, origin_col] == id]
                unique_days, counts_per_day = np.unique(shu_moves[:, date_moved_col], return_counts=True)
                shus_max_moves[i] = counts_per_day.max() if len(counts_per_day) > 0 else 0
            max_moves_shus.append(shus_max_moves)
            # ssm_cols = [origin_col, date_moved_col, pat_grp_col]
            # batch_move, num_in_batch = np.unique(steady_state_moves[:, ssm_cols], axis=0, return_counts=True)
        mean_max_moves_shus = np.mean(max_moves_shus, axis=0)
        std_err_max_moves_shus = np.std(max_moves_shus, axis=0) / np.sqrt(len(max_moves_shus))
        mean_max_moves = np.mean(np.max(max_moves_shus, axis=1))
        std_err_max_moves = np.std(np.max(max_moves_shus, axis=1)) / np.sqrt(len(max_moves_shus))
        results.update({_f: {'mean_max_each_shu': mean_max_moves_shus,
                           'std_err_max_each_shu': std_err_max_moves_shus,
                           'mean_max_overall': mean_max_moves,
                           'std_err_max_overall': std_err_max_moves,
                           'shu_names': names,}})
        data.close()
    return results


def plot_dist_moves_per_day1(files: dict[str, str], shus: dict, warm_up = 0, move_reason=3, file_labels=None,
                            origin_col=3, date_moved_col=6, pat_grp_col=8, figsize=(10,6), dpi=200, pdf=None,
                            plot_type='violin', jitter=True, dodge=True, 
                            legend_title='Scenario',):
    """
    Loads move data from npz files, filters it, and plots the distribution
    of daily moves per SHU in a 5x3 grid. The central plot shows all SHUs
    combined, and the 14 surrounding plots each show an individual SHU.
    """
    ids = shus.keys()
    names = list(shus.values())
    data_for_df = []
    for f_key in files.keys():
        f = files[f_key]
        if f.endswith('_output.tsv'):
            f = f.replace('_output.tsv', '_moves.npz')
        data = np.load(f)
        for replication in data.files:
            moves = data[replication]
            steady_state_moves = moves[moves[:, date_moved_col] > warm_up]
            if type(move_reason) == int:
                steady_state_moves = steady_state_moves[steady_state_moves[:, pat_grp_col] == move_reason]
            for i, id in enumerate(ids):
                shu_name = shus[id]
                shu_moves = steady_state_moves[steady_state_moves[:, origin_col] == id]
                if shu_moves.shape[0] == 0:
                    continue
                unique_days, counts_per_day = np.unique(shu_moves[:, date_moved_col], return_counts=True)
                scenario_name = f_key if file_labels is None else file_labels[f_key]
                data_to_append = [{'daily_moves': count, 'scenario': scenario_name, 'shu': shu_name} for count in counts_per_day]
                data_for_df.extend(data_to_append)
        data.close()
    df = pd.DataFrame(data_for_df)
    
    # --- 2. Setup the 5x3 Plotting Grid ---
    # Create a Matplotlib Figure and Axes
    fig, axes = plt.subplots(5, 3, figsize=figsize, dpi=dpi)
    # Define the center plot position
    middle_row, middle_col = 2, 1
    
    shu_names_list = list(shus.values())
    shu_iter = iter(shu_names_list)
    
    handles, labels = [], [] # For storing legend items
    
    # --- 3. Iterate through the grid and create plots ---
    for r in range(5):
        for c in range(3):
            ax = axes[r][c]
            # --- 3a. The Middle "Combined" Plot ---
            if r == middle_row and c == middle_col:
                if plot_type == 'violin':
                    sns.violinplot(data=df, x='daily_moves', y='scenario',
                        hue='scenario', inner='point', palette='Set2', ax=ax, legend=True,)
                elif plot_type == 'strip':
                    sns.stripplot(data=df, x='daily_moves', y='scenario', hue='scenario',
                        jitter=jitter, dodge=dodge, alpha=0.7, palette='Set2', ax=ax, legend=True,)
                ax.set_title('All SHUs', fontsize=14, weight='bold')
                ax.set_xlabel('Number of Units Transferred')
                ax.set_ylabel('')
                ax.grid(axis='x', linestyle='--', alpha=0.7)
                # Get legend handles from this plot
                handles, labels = ax.get_legend_handles_labels()
                ax.legend().remove() # Remove small subplot legend
            
            # --- 3b. Individual SHU Plots ---
            else:
                try:
                    shu_name = next(shu_iter)
                    shu_df = df[df['shu'] == shu_name]
                    
                    if plot_type == 'violin':
                        sns.violinplot(data=shu_df, x='daily_moves', y='scenario', # Flipped y-axis
                            hue='scenario', # Use hue for consistent color
                            inner='point', palette='Set2', ax=ax, legend=False)
                    elif plot_type == 'strip':
                        sns.stripplot(data=shu_df, x='daily_moves', y='scenario', # Flipped y-axis
                            hue='scenario', # Use hue for consistent color
                            jitter=jitter, dodge=dodge, # No dodging for individual plots
                            alpha=0.7, palette='Set2', ax=ax,legend=False)
                    ax.set_title(shu_name, fontsize=12)
                    ax.set_xlabel('Number of Units Transferred')
                    ax.set_ylabel('') # Remove "scenario" label for clarity
                    ax.grid(axis='x', linestyle='--', alpha=0.7)
                except StopIteration:
                    # Ran out of SHUs (e.g., if you have < 14)
                    # Or this is the 15th slot
                    ax.axis('off')
                    
    # --- 4. Final Figure-Level Touches ---
    # Add a single, shared legend to the figure
    if handles:
        fig.legend(handles, labels, title=legend_title, 
                   bbox_to_anchor=(0.98, 0.65), loc='upper left',)
                #    title_fontsize='14', fontsize='12')
    # Add a main title
    fig.suptitle('Distribution of Daily Transfers per SHU across Scenarios', fontsize=20, weight='bold')
    # Adjust layout to prevent overlap and make room for titles/legend
    fig.tight_layout(rect=[0, 0, 0.95, 0.99]) # rect=[left, bottom, right, top]
    
    if pdf is not None:
        fig.savefig(pdf, bbox_inches='tight')
    fig.show()
    

def plot_dist_moves_per_day(files: dict[str, str], shus: dict, warm_up=0, move_reason=3, file_labels=None,
                            origin_col=3, date_moved_col=6, pat_grp_col=8, figsize=(20, 25), dpi=200, pdf=None,
                            plot_type='violin', jitter=True, dodge=True,
                            legend_title='Scenario', separate_plots=False):
    """
    Loads move data from npz files, filters it, and plots the distribution
    of daily moves.

    By default (separate_plots=False), plots a 5x3 grid:
    - The central plot shows all SHUs combined.
    - The 14 surrounding plots each show an individual SHU.

    If separate_plots=True, generates 15 separate figures (one for each SHU
    and one for 'All SHUs'). If pdf is provided, these are saved
    as a single multi-page PDF.

    Args:
        files (dict): Dictionary of scenario names to file paths.
        shus (dict): Dictionary of SHU IDs to SHU names.
        warm_up (int): Day to start analysis from.
        move_reason (int): Patient group/move reason to filter for.
        file_labels (dict, optional): Alternative labels for scenarios.
        origin_col (int): Column index for origin SHU ID.
        date_moved_col (int): Column index for move date.
        pat_grp_col (int): Column index for patient group/move reason.
        figsize (tuple): Figure size. Default is (20, 25) for grid,
                         recommend (10, 6) for separate plots.
        dpi (int): Figure resolution.
        pdf (str, optional): Path to save the plot. If separate_plots=True,
                                this will be a multi-page PDF.
        plot_type (str): 'violin', 'strip', or 'cdf'.
        jitter (bool): For strip plots.
        dodge (bool): For strip plots.
        legend_title (str): Title for the legend.
        separate_plots (bool): If True, generate separate plots instead of a grid.
    """
    ids = shus.keys()
    names = list(shus.values())
    data_for_df = []
    
    for f_key in files.keys():
        f = files[f_key]
        if f.endswith('_output.tsv'):
            f = f.replace('_output.tsv', '_moves.npz')
        data = np.load(f)
        for replication in data.files:
            moves = data[replication]
            steady_state_moves = moves[moves[:, date_moved_col] > warm_up]
            if type(move_reason) == int:
                steady_state_moves = steady_state_moves[steady_state_moves[:, pat_grp_col] == move_reason]
            for i, id in enumerate(ids):
                shu_name = shus[id]
                shu_moves = steady_state_moves[steady_state_moves[:, origin_col] == id]
                if shu_moves.shape[0] == 0:
                    continue
                unique_days, counts_per_day = np.unique(shu_moves[:, date_moved_col], return_counts=True)
                scenario_name = f_key if file_labels is None else file_labels.get(f_key, f_key)
                data_to_append = [{'daily_moves': count, 'scenario': scenario_name, 'shu': shu_name} for count in counts_per_day]
                data_for_df.extend(data_to_append)
        data.close()
      
    df = pd.DataFrame(data_for_df)
    shu_names_list = list(shus.values())

    # ------------------------------------------------------------------
    # --- Option 1: Plot as a single 5x3 Grid (default)
    # ------------------------------------------------------------------
    if not separate_plots:
        fig, axes = plt.subplots(5, 3, figsize=figsize, dpi=dpi)
        middle_row, middle_col = 2, 1
        shu_iter = iter(shu_names_list)
        handles, labels = [], []
        for r in range(5):
            for c in range(3):
                ax = axes[r][c]
                # --- 1a. The Middle "Combined" Plot ---
                if r == middle_row and c == middle_col:
                    if plot_type == 'violin':
                        sns.violinplot(data=df, x='daily_moves', y='scenario',
                            hue='scenario', inner='point', palette='Set2', ax=ax, legend=True)
                    elif plot_type == 'strip':
                        sns.stripplot(data=df, x='daily_moves', y='scenario', hue='scenario',
                            jitter=jitter, dodge=dodge, alpha=0.7, palette='Set2', ax=ax, legend=True)
                    elif plot_type == 'cdf':
                        sns.ecdfplot(data=df, x='daily_moves', hue='scenario',
                            palette='Set2', ax=ax, legend=True)
                    ax.set_title('All SHUs', fontsize=14, weight='bold')
                    ax.set_xlabel('Number of Units Transferred')
                    ax.set_ylabel('')
                    ax.grid(axis='x', linestyle='--', alpha=0.7)
                    if plot_type == 'cdf':
                        ax.set_ylabel('Cumulative Probability')
                        handles = ax.get_legend().legend_handles
                        labels = [text.get_text() for text in ax.get_legend().texts]
                        ax.legend().remove()
                    else:
                        handles, labels = ax.get_legend_handles_labels()
                        ax.legend().remove()
                # --- 1b. Individual SHU Plots ---
                else:
                    try:
                        shu_name = next(shu_iter)
                        shu_df = df[df['shu'] == shu_name]
                        if plot_type == 'violin':
                            sns.violinplot(data=shu_df, x='daily_moves', y='scenario',
                                hue='scenario', inner='point', palette='Set2', ax=ax, legend=False)
                        elif plot_type == 'strip':
                            sns.stripplot(data=shu_df, x='daily_moves', y='scenario',
                                hue='scenario', jitter=jitter, dodge=dodge,
                                alpha=0.7, palette='Set2', ax=ax, legend=False)
                        elif plot_type == 'cdf':
                            sns.ecdfplot(data=shu_df, x='daily_moves', hue='scenario',
                                palette='Set2', ax=ax, legend=False)
                        ax.set_title(shu_name, fontsize=12)
                        ax.set_xlabel('Number of Units Transferred')
                        ax.set_ylabel('')
                        ax.grid(axis='x', linestyle='--', alpha=0.7)
                        if plot_type == 'cdf':
                            ax.set_ylabel('Cumulative Probability')
                    except StopIteration:
                        ax.axis('off')
                        
        if handles:
            fig.legend(handles, labels, title=legend_title, 
                       bbox_to_anchor=(0.98, 0.65), loc='upper left')
        fig.suptitle('Distribution of Daily Transfers per SHU across Scenarios', fontsize=20, weight='bold')
        fig.tight_layout(rect=[0, 0, 0.95, 0.99])
        if pdf:
            fig.savefig(pdf, bbox_inches='tight')
        fig.show()

    # ------------------------------------------------------------------
    # --- Option 2: Plot 15 Separate Figures
    # ------------------------------------------------------------------
    else:
        # Helper function to create a single plot
        def _plot_ax(ax, data_to_plot, title_str):
            if plot_type == 'violin':
                sns.violinplot(data=data_to_plot, x='daily_moves', y='scenario',
                    hue='scenario', inner='point', palette='Set2', ax=ax, legend=True)
            elif plot_type == 'strip':
                sns.stripplot(data=data_to_plot, x='daily_moves', y='scenario', hue='scenario',
                    jitter=jitter, dodge=dodge, alpha=0.7, palette='Set2', ax=ax, legend=True)
            elif plot_type == 'cdf':
                sns.ecdfplot(data=data_to_plot, x='daily_moves', hue='scenario',
                    palette='Set2', ax=ax,)# legend=True)
            ax.set_title(title_str, fontsize=16, weight='bold')
            ax.set_xlabel('Number of Units Transferred')
            if plot_type == 'cdf':
                ax.set_ylabel('Cumulative Probability')
                # ax.legend(title=legend_title)
            else:
                ax.set_ylabel('')
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles, labels, title=legend_title)
            ax.grid(axis='x', linestyle='--', alpha=0.7)

        # Setup multi-page PDF if path is given
        pdf_doc = None
        if pdf is not None:
            pdf_doc = PdfPages(pdf)

        # 1. "All SHUs" Plot
        fig_all, ax_all = plt.subplots(figsize=figsize, dpi=dpi)
        _plot_ax(ax_all, df, 'All SHUs')
        fig_all.tight_layout()
        if pdf_doc is not None:
            pdf_doc.savefig(fig_all)
        else:
            fig_all.show()
        # plt.close(fig_all)

        # 2. Individual SHU Plots
        for shu_name in shu_names_list:
            shu_df = df[df['shu'] == shu_name]
            if shu_df.empty:
                continue
            fig_shu, ax_shu = plt.subplots(figsize=figsize, dpi=dpi)
            _plot_ax(ax_shu, shu_df, shu_name)
            fig_shu.tight_layout()
            if pdf_doc is not None:
                pdf_doc.savefig(fig_shu)
            else:
                fig_shu.show()
            # plt.close(fig_shu)

        if pdf_doc is not None:
            pdf_doc.close()


def calc_courier_costs_binned(files: Union[list[str], str], cost_matrix: str | pd.DataFrame, shus: dict,
                              warm_up = 0, move_reason=3, dest_col=9, file_keys=None,
                              order_of_shus=None, pat_grp_col=8, origin_col=3, date_moved_col=6,):
    if isinstance(files, str):
        files = [files]
    if file_keys is None:
        file_keys = files
    ids = shus.keys()
    names = shus.values()
    all_shu_combinations = list_of_permutations([tuple(ids), tuple(ids)])
    ids_to_index = {id: i for i, id in enumerate(ids)}
    results = {}
    for i, f in enumerate(tqdm.tqdm(files, desc='Files', position=0)):
        if f.endswith('_output.tsv'):
            f = f.replace('_output.tsv', '_moves.npz')
        data = np.load(f)
        all_costs = []
        
        for replication in data.files:
            moves = data[replication]
            steady_state_moves = moves[moves[:, date_moved_col] > warm_up]
            if type(move_reason) == int:
                steady_state_moves = steady_state_moves[steady_state_moves[:, pat_grp_col] == move_reason]
            # ssm_cols = [origin_col, date_moved_col, pat_grp_col, dest_col]
            batch_move, num_in_batch = np.unique(steady_state_moves[:, [origin_col, date_moved_col, dest_col]], axis=0, return_counts=True)
            all_batch_moves, num_days = np.unique(batch_move[:, [0, 2]], axis=0, return_counts=True)
            costs = np.zeros((max(ids)+1, max(ids)+1))
            costs[all_batch_moves[:, 0], all_batch_moves[:, 1]] = num_days
            all_costs.append(costs)
        all_costs_mean = np.mean(all_costs, axis=0)
        all_costs_std_err = np.std(all_costs, axis=0) / np.sqrt(len(data.files))
        all_costs_mean = all_costs_mean[list(ids), :][:, list(ids)]
        all_costs_std_err = all_costs_std_err[list(ids), :][:, list(ids)]
        if isinstance(cost_matrix, str):
            cost_matrix = pd.read_csv(cost_matrix, sep='\t', index_col=0)
        costs_mean_df = pd.DataFrame(all_costs_mean, index=names, columns=names) * cost_matrix
        costs_std_err_df = pd.DataFrame(all_costs_std_err, index=names, columns=names) * cost_matrix
        results.update({file_keys[i]: (costs_mean_df.iloc[order_of_shus, order_of_shus],
                                                    costs_std_err_df.iloc[order_of_shus, order_of_shus])})
        data.close()
    return results
        

def plot_courier_costs(dfs: dict, titles: list[str], annotate=True, cmap='YlGnBu',
                       figsize=(10, 8), pdf=None, dpi=200, fmt=',.0f', divide_by=1):
    for f, title in zip(dfs.keys(), titles):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        sns.heatmap(dfs[f] / divide_by, ax=ax, cmap=cmap, annot=annotate, fmt=fmt)
        ax.set_title(title)
        ax.set_xlabel('Destination SHU')
        ax.set_ylabel('Source SHU')
        if pdf is not None:
            fig.savefig(pdf, bbox_inches='tight')
        fig.show()


def list_of_permutations(domain_list) -> list:
    prototype = []
    num_permutations = 1
    divisors = []
    len_dl = len(domain_list)
    for i in range(len_dl - 1, -1, -1):
        domain = domain_list[i]
        prototype = prototype + [domain[0]]
        len_d = len(domain)
        num_permutations = num_permutations * len_d
        divisors.append(num_permutations / len_d)
    permutation_list = []
    for i in range(num_permutations):
        permutation = []
        for j in range(len_dl):
            domain = domain_list[j]
            k = int(i / divisors[(len_dl - 1 - j)]) % len(domain)
            permutation.append(domain[k])
        permutation_list.append(permutation)
    return permutation_list


def histogram_time_unmet_requests(datafilename, figsize=(8, 6), dpi=200, pdf=None, title=None):
    if datafilename.endswith('_output.tsv'):
        datafilename = datafilename.replace('_output.tsv', '_unmet_requests.npz')
    data = np.load(os.path.realpath(os.path.expanduser(datafilename)))
    time_of_unmet_requests =[]
    for replication in data.files:
        times = data[replication][:, 3]
        time_of_unmet_requests.append(times)
    time_of_unmet_requests = np.concatenate(time_of_unmet_requests)
    counts, bins = np.histogram(time_of_unmet_requests, bins=np.arange(1, 7*6*4+ 7, 7))
    counts_avg = counts / len(data.files)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    plt.stairs(counts_avg, bins, fill=True, color='gray', alpha=0.5)
    draw_warmup_line(ax, color='black', x=7*6*3+0.5)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Average Number of Unmet Requests')
    ax.set_title(title)
    fig.tight_layout()
    if pdf is not None:
        fig.savefig(pdf, bbox_inches='tight')
    fig.show()
    data.close()


def histogram_time_unmet_requests_amounts(datafilename, figsize=(8, 6), dpi=200, pdf=None, title=None,
                                          pat_groups=None):
    if datafilename.endswith('_output.tsv'):
        datafilename = datafilename.replace('_output.tsv', '_unmet_requests.npz')
    data = np.load(os.path.realpath(os.path.expanduser(datafilename)))
    counts = np.arange(0, 7*6*4+1)
    time_of_unmet_requests =[]
    for replication in data.files:
        units_dates_grps = data[replication][:, [2,3,4]]
        time_of_unmet_requests.append(units_dates_grps)
    time_of_unmet_requests = np.vstack(time_of_unmet_requests)
    all_times = time_of_unmet_requests[:, 1]
    unique_times = np.unique(all_times)
    for time in unique_times:
        time_indices = all_times == time
        time_units_dates_grps = time_of_unmet_requests[time_indices]
        time_units = time_units_dates_grps[:, 0]
        time_grps = time_units_dates_grps[:, 2]
        if pat_groups is not None:
            grps_indices = time_grps == pat_groups
            time_units = time_units[grps_indices]
        # time_counts = np.zeros(7*6*4+1)
        # for grp in pat_groups.keys():
        #     grp_indices = time_grps == grp
        #     grp_units = time_units[grp_indices]
        #     grp_counts, _ = np.histogram(grp_units, bins=np.arange(0, 7*6*4+1))
        #     time_counts += grp_counts
        # counts = np.vstack([counts, time_counts])
        counts[time] = time_units.sum() / len(data.files)
    # time_of_unmet_requests = np.concatenate(time_of_unmet_requests)
    counts = counts[1:]
    bins = np.arange(1, 7*6*4+ 2)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.stairs(counts, bins, fill=True, color='gray', alpha=0.5)
    draw_warmup_line(ax, color='black', x=7*6*3+0.5)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Average Number of Units Short')
    ax.set_title(title)
    fig.tight_layout()
    if pdf is not None:
        fig.savefig(pdf, bbox_inches='tight')
    fig.show()
    data.close()
   

def distribution_multiple_moves(datafilename, figsize=(8,6), dpi=200, pdf=None, title=None,
                                warmup=7*6*3, move_reason=None, xlim=None, ylim=None,
                                watch_antigen=0, watch_phenotype=0, date_moved_col=6,
                                pat_grp_col=8, phenotype_col=1, unit_id_col=0, shelf_life=35,
                                plot_graph=True, return_hist=False,
                                fontsize=None,):
    if datafilename.endswith('_output.tsv'):
        datafilename = datafilename.replace('_output.tsv', '_moves.npz')
    data = np.load(os.path.realpath(os.path.expanduser(datafilename)))
    all_moves = []
    max_move = 10
    min_move = 1
    all_rep_counts = []
    rep_bins = np.arange(0, shelf_life+1)
    for replication in data.files:
        moves = data[replication]
        moves = moves[moves[:, date_moved_col] > warmup]
        all_other_moves = np.empty((0, moves.shape[1]), dtype=moves.dtype)
        if move_reason is not None:
            moves_for_reason_indicator = moves[:, pat_grp_col] == move_reason
            all_other_moves = moves[~moves_for_reason_indicator]
            moves = moves[moves_for_reason_indicator]
        moves = moves[moves[:, phenotype_col] & watch_antigen == watch_phenotype]
        moves_id = moves[:, unit_id_col]
        unique_ids, counts = np.unique(moves_id, return_counts=True)
        max_move = max(max_move, counts.max() if len(counts) > 0 else 0)
        all_moves.append(counts)
        # moves.append(data[replication])
        rep_counts, _ = np.histogram(counts, bins=rep_bins)
        all_rep_counts.append(rep_counts)
    all_counts, bins = np.histogram(np.concatenate(all_moves, dtype=int), bins=np.arange(0, max_move+1))
    all_rep_mean_counts = np.mean(all_rep_counts, axis=0)
    all_rep_std_err_counts = np.std(all_rep_counts, axis=0) / np.sqrt(len(all_rep_counts))
    if plot_graph:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        # ax.stairs(all_counts[1:]/len(data.files), bins.astype(int)[1:], fill=True, color='gray', alpha=0.5)
        # ax.stairs(all_counts/len(data.files), bins - 0.5, fill=True, color='gray', alpha=0.5)
        ax.stairs(all_rep_mean_counts, rep_bins - 0.5, fill=True, color='gray', alpha=0.75)
        ax.stairs(all_rep_mean_counts + all_rep_std_err_counts, rep_bins - 0.5, fill=False, color='black', alpha=0.4, linestyle='--')
        ax.stairs(all_rep_mean_counts - all_rep_std_err_counts, rep_bins - 0.5, fill=False, color='black', alpha=0.4, linestyle='--')
        # draw_warmup_line(ax, color='black', x=7*6*3+0.5)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xticks(np.arange(max_move+1))
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%d'))
        ax.set_xlabel('Number of Transfers', fontsize=fontsize)
        ax.set_ylabel('Frequency', fontsize=fontsize)
        ax.set_title(title, fontsize=fontsize)
        fig.tight_layout()
        if pdf is not None:
            fig.savefig(pdf, bbox_inches='tight')
        fig.show()
    data.close()
    if return_hist:
        return all_rep_mean_counts, all_rep_std_err_counts
    
    # fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    # sns.histplot(moves, ax=ax, bins=np.arange(0, 7*6*4+1))
    # draw_warmup_line(ax, color='black', x=7*6*3+0.5)
    # ax.set_xlabel('Number of Moves')
    # ax.set_ylabel('Frequency')
    # ax.set_title(title)
    # fig.tight_layout()
    # if pdf is not None:
    #     fig.savefig(pdf, bbox_inches='tight')
    # fig.show()
    # data.close()
    
def plot_transfer_stairs_pyramid(
    file_left,
    file_right,
    label_left="E02: Zero transportation penalty",
    label_right="E03: Transportation penalty",
    warmup=0,
    move_reason=None,
    shelf_life=23,
    drop_zero_bin=True,
    show_uncertainty=True,
    figsize=(11, 6),
    dpi=220,
    fontsize=18,
    color_left="#e66101",    # orange
    color_right="#5e3c99",   # purple
    alpha_fill=0.70,
    pdf=None,
    title=None,
    xlim=None,
    ylim=None,
):
    # Get mean and standard error histograms from your existing helper
    mean_l, se_l = distribution_multiple_moves(
        file_left,
        plot_graph=False,
        return_hist=True,
        warmup=warmup,
        move_reason=move_reason,
        shelf_life=shelf_life,
    )
    mean_r, se_r = distribution_multiple_moves(
        file_right,
        plot_graph=False,
        return_hist=True,
        warmup=warmup,
        move_reason=move_reason,
        shelf_life=shelf_life,
    )

    # Bin indices correspond to number of transfers
    transfer_counts = np.arange(len(mean_l))

    if drop_zero_bin:
        transfer_counts = transfer_counts[1:]
        mean_l, se_l = mean_l[1:], se_l[1:]
        mean_r, se_r = mean_r[1:], se_r[1:]

    # Build bin edges for stairs (one edge longer than values)
    edges = np.r_[transfer_counts - 0.5, transfer_counts[-1] + 0.5]

    # Mirror left side (negative x)
    left_mean = -mean_l
    right_mean = mean_r

    # Uncertainty envelopes (clipped at zero)
    left_lo_mag = np.clip(mean_l - se_l, 0, None)
    left_hi_mag = mean_l + se_l
    right_lo = np.clip(mean_r - se_r, 0, None)
    right_hi = mean_r + se_r

    left_lo = -left_lo_mag
    left_hi = -left_hi_mag

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Filled stairs silhouettes
    ax.stairs(left_mean, edges, orientation="horizontal", baseline=0,
                fill=True, color=color_left, alpha=alpha_fill, label=label_left)
    ax.stairs(right_mean, edges, orientation="horizontal", baseline=0,
                fill=True, color=color_right, alpha=alpha_fill, label=label_right)

    # Main step outlines
    ax.stairs(left_mean, edges, orientation="horizontal", baseline=0,
                fill=False, color=color_left, linewidth=1.8)
    ax.stairs(right_mean, edges, orientation="horizontal", baseline=0,
                fill=False, color=color_right, linewidth=1.8)

    # Dashed step uncertainty bounds
    if show_uncertainty:
        ax.stairs(left_lo, edges, orientation="horizontal", baseline=0,
                    fill=False, color=color_left, linestyle="--", alpha=0.55, linewidth=1.2)
        ax.stairs(left_hi, edges, orientation="horizontal", baseline=0,
                    fill=False, color=color_left, linestyle="--", alpha=0.55, linewidth=1.2)
        ax.stairs(right_lo, edges, orientation="horizontal", baseline=0,
                    fill=False, color=color_right, linestyle="--", alpha=0.55, linewidth=1.2)
        ax.stairs(right_hi, edges, orientation="horizontal", baseline=0,
                    fill=False, color=color_right, linestyle="--", alpha=0.55, linewidth=1.2)

    # Central divider
    ax.axvline(0, color="black", linewidth=1.0, alpha=0.8)

    # Format axis so both sides show absolute frequency
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: f"{abs(int(x)):,}"))

    # Symmetric x-limits
    auto_xmax = max(np.max(mean_l + se_l), np.max(mean_r + se_r)) * 1.12
    if xlim is None:
        ax.set_xlim(-auto_xmax, 2000)
    else:
        ax.set_xlim(*xlim)

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_yticks(transfer_counts)
    ax.set_xlabel("Frequency", fontsize=fontsize)
    ax.set_ylabel("Number of Transhipments", fontsize=fontsize)
    if title is not None:
        ax.set_title(title, fontsize=fontsize)

    ax.legend(loc="upper center", ncol=2, frameon=False, fontsize=fontsize * 0.85,
              bbox_to_anchor=(0.835, 1),)
    ax.grid(axis="x", alpha=0.18)
    fig.tight_layout()

    if pdf is not None:
        fig.savefig(pdf, bbox_inches="tight")

    plt.show()
    return fig, ax


  
def contribution_of_matching_penalties(filename: str):
    if filename.endswith('_output.tsv'):
        filename = filename.replace('_output.tsv', '_penalty_contributions.npz')
    data = np.load(os.path.realpath(os.path.expanduser(filename)))
    all_penalties = []
    for replication in data.files:
        penalties = data[replication]
        all_penalties.append(penalties)
    all_penalties = np.vstack(all_penalties)
    data.close()
    return all_penalties
 
# def contribution_of_matching_penalties(datafilename: str, warmup=7*6*3, ):
#     if datafilename.endswith('_output.tsv'):
#         datafilename = datafilename.replace('_output.tsv', '_matches.npz')
#     data = np.load(os.path.realpath(os.path.expanduser(datafilename)))
#     antigen = antigens()
#     all_components = {}
#     all_matches = []
#     for replication in data.files:
#         matches = data[replication]
#         matches = matches[matches[:, 2] > warmup]
#         donor_phen_vectors = antigen.convert_to_binarray(matches[:, 4])
#         patient_phen_vectors = antigen.convert_to_binarray(matches[:, 3])
#         # fifo, mismatching, transport
#     ...


def plot_locations_of_expiries(filename: str, shus: dict,
                               figsize=(10, 6), dpi=200, pdf=None, title=None,
                               location_col=3, ylim=None,):
    if filename.endswith('_output.tsv'):
        filename = filename.replace('_output.tsv', '_expiry_records.npz')
    data = np.load(os.path.realpath(os.path.expanduser(filename)))
    locs_and_counts = np.empty((0, 2), dtype=int)
    for replication in data.files:
        expiries = data[replication]
        unique_locs, counts = np.unique(expiries[:, location_col], return_counts=True)
        locs_and_counts_rep = np.vstack((unique_locs, counts)).T
        locs_and_counts = np.vstack((locs_and_counts, locs_and_counts_rep))
    locs_sum = [locs_and_counts[:,1].sum(axis=0, where=locs_and_counts[:,0]==i) for i in shus.keys()]
    locs_sum = np.array(locs_sum)
    locs_avg = locs_sum / len(data.files)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.bar([shus[i] for i in shus.keys()], locs_avg, color='gray', alpha=0.7)
    ax.set_xticks(np.arange(len(shus)))
    ax.set_xticklabels([shus[i].split('_')[-1] for i in shus.keys()], rotation=45, ha='right')
    ax.set_xlabel('SHU Location')
    ax.set_ylabel('Average Number of Expiries')
    ylim = ylim if ylim is not None else (None, None)
    ax.set_ylim(*ylim)
    ax.set_title(title)
    fig.tight_layout()
    if pdf is not None:
        fig.savefig(pdf, bbox_inches='tight')
    fig.show()
    
    data.close()
    
    
def plot_times_of_expiries(filename: str,
                            figsize=(8, 6), dpi=200, pdf=None, title=None,
                            date_col=2, time_horizon=7*6*4, warmup=7*6*3, shelf_life= 35,):
    if filename.endswith('_output.tsv'):
        filename = filename.replace('_output.tsv', '_expiry_records.npz')
    data = np.load(os.path.realpath(os.path.expanduser(filename)))
    all_dates = []
    for replication in data.files:
        expiries = data[replication]
        dates = expiries[:, date_col] + shelf_life - 1 - warmup
        all_dates.append(dates)
    all_dates = np.concatenate(all_dates)
    counts, bins = np.histogram(all_dates, bins=np.arange(1, time_horizon - warmup + 2))
    counts_avg = counts / len(data.files)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.stairs(counts_avg, bins, fill=True, color='gray', alpha=0.5)
    # draw_warmup_line(ax, color='black', x=warmup+0.5)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Average Number of Expiries')
    ax.set_title(title)
    ax.grid()
    fig.tight_layout()
    if pdf is not None:
        fig.savefig(pdf, bbox_inches='tight')
    fig.show()
    data.close()


def how_many_expiries_during_transit(filename: str, shelf_life=35,
                                     transit_col=5, time_horizon=7*6*4,):
    if filename.endswith('_output.tsv'):
        filename = filename.replace('_output.tsv', '_expiry_records.npz')
    data = np.load(os.path.realpath(os.path.expanduser(filename)))
    expiries_in_transit = []
    all_expiries = []
    for replication in data.files:
        expiries = data[replication]
        # expiries = expiries[expiries[:, date_col] > warmup]
        time_left_in_transit = expiries[:, transit_col]
        expiries_during_transit = expiries[(time_left_in_transit > 0)] #& (transit_times < shelf_life)]
        expiries_in_transit.append(len(expiries_during_transit))
        all_expiries.append(len(expiries))
    expiries_in_transit = np.mean(expiries_in_transit)
    all_expiries = np.mean(all_expiries)
    data.close()
    return expiries_in_transit, all_expiries


def plot_age_distributions_locations(datafilename, age_range=None, days_range=None,
                                     figsize=(12, 18), dpi=200, pdf=None,
                                     title=None, shelf_life=35, time_horizon=7*6*4,
                                     error_over_days=False,):
    if datafilename.endswith('_output.tsv'):
        datafilename = datafilename.replace('_output.tsv', '_age_distributions.npz')
    data = np.load(os.path.realpath(os.path.expanduser(datafilename)))
    age_dist = data['location_age_dists']
    age_dist_std_err = data['location_age_dists_stderr']
    loc_names = data['locations']
    data.close()
    # age_dist = get_age_dist(datafilename, 'location_age_dists')
    if age_range is None:
        age_range = np.arange(0, shelf_life + 1)
    if days_range is None:
        days_range = np.arange(1, time_horizon + 1)
    arr = age_dist[days_range-1,:, :]
    arr = arr[:,:,age_range]
    arr_stderr = age_dist_std_err[days_range-1,:, :]
    arr_stderr = arr_stderr[:,:,age_range]
    arr_mean = arr.mean(axis=0)
    arr_stderr_mean = (arr_stderr ** 2).sum(axis=0) ** 0.5 / arr.shape[0]
    arr_std_err = arr.std(axis=0) / np.sqrt(arr.shape[0])
    arr_stderr_mean = arr_std_err if error_over_days else arr_stderr_mean
    fig, axes = plt.subplots(7, 2, figsize=figsize, dpi=dpi)
    for i, ax in enumerate(axes.flatten()):
        ax.stairs(arr_mean[i, :], np.arange(age_range.min(), age_range.max()+2) - 0.5,
                  fill=True, color='gray', alpha=1, label='Mean')
        ax.stairs(arr_mean[i, :] + arr_stderr_mean[i, :], np.arange(age_range.min(), age_range.max()+2) - 0.5,
                  fill=False, color='black', alpha=0.35, ls=':', label='Std. Err.')
        ax.stairs(arr_mean[i, :] - arr_stderr_mean[i, :], np.arange(age_range.min(), age_range.max()+2) - 0.5,
                  fill=False, color='black', alpha=0.35, ls=':')
        ax.set_title(loc_names[i])
        ax.set_xlabel('Age (days)')
        ax.set_ylabel('Average Units')
        # ax.legend(frameon=False)
    fig.suptitle(title)
    fig.legend(['Mean', 'Std. Err.'], loc='upper right')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if pdf is not None:
        fig.savefig(pdf, bbox_inches='tight')
    fig.show()


def plot_objective_means_pareto(file_list, objective1='alloimmunisations', objective2='scd_moves',
                                labels=None, label_points=False,
    show_errorbars=True, figsize=(8, 6), title=None, pdf=None, dpi=200,
    legend_ncol=None, legend_fontsize=12, legend_title_fontsize=18, fontsize=None,
    palette='tab20', palette_offset=0,
    marker_cycle=('o', 's', '^', 'D', 'v', 'P', 'X', '<', '>', '*', 'h', '8'),
    use_marker_variation=False, marker_offset=0, marker_size=8,):
    """
    Reads a list of objective files, computes the mean and SEM of two objectives
    (defaults to 'alloimmunisation' and 'scd_moves' columns), and plots them with optional error bars.
    
    You can either show labels on the plot or in a legend (not both).
    
    Parameters
    ----------
    file_list : list[str or Path]
        List of paths to TSV files.
    objective1 : str, optional (default='alloimmunisation')
        Column name for the Y-axis objective.
    objective2 : str, optional (default='scd_moves')
        Column name for the X-axis objective.
    labels : list[str], optional
        Labels for each file, must be the same length as file_list.
        If None, filenames (without extensions) will be used.
    label_points : bool, optional (default=False)
        If True, annotate points with labels directly on the plot.
        If False, show labels in a legend.
    show_errorbars : bool, optional (default=True)
        If True, display error bars based on SEM of both variables.
    figsize : tuple(float, float), optional (default=(8, 6))
        Figure size in inches.
    title : str, optional
        Title for the plot. If None, no title is shown.
    pdf : str or Path, optional
        If provided, saves the figure as a PDF to this path.
    palette : str or sequence, optional
        Matplotlib/seaborn palette name or an explicit sequence of colors.
        Defaults to 'tab20' to avoid early color repetition when plotting many scenarios.
    palette_offset : int, optional (default=0)
        Rotates color assignment by this many positions.
    marker_cycle : sequence[str], optional
        Marker symbols to cycle through when `use_marker_variation` is True.
    use_marker_variation : bool, optional (default=False)
        If True, cycles marker symbols in addition to colors to improve accessibility.
    marker_offset : int, optional (default=0)
        Rotates marker assignment by this many positions.
    marker_size : float, optional (default=8)
        Marker size passed to matplotlib.

    Returns
    -------
    None
    """
    _file_list = [datafilename.replace('_output.tsv', '_objectives.tsv') if datafilename.endswith('_output.tsv') else datafilename for datafilename in file_list]
    file_list = [os.path.realpath(os.path.expanduser(f)) for f in _file_list]

    if labels is None:
        labels = [f'{i}a' for i in range(len(file_list))]
    elif len(labels) != len(file_list):
        raise ValueError("`labels` must be the same length as `file_list`.")

    if isinstance(palette, str):
        colours = sns.color_palette(palette, n_colors=max(len(file_list), 1))
    else:
        colours = list(palette)
        if len(colours) == 0:
            raise ValueError("`palette` must contain at least one color.")

    markers = list(marker_cycle)
    if use_marker_variation and len(markers) == 0:
        raise ValueError("`marker_cycle` must contain at least one marker when marker variation is enabled.")

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi,)
    
    # Compute and plot for each file
    for i, (file_path, label) in enumerate(zip(file_list, labels)):
        df = pd.read_csv(file_path, sep='\t')

        mean_allo = df[objective1].mean()
        mean_scd = df[objective2].mean()
        sem_allo = df[objective1].sem(ddof=0)
        sem_scd = df[objective2].sem(ddof=0)

        colour = colours[(i + palette_offset) % len(colours)]
        marker = markers[(i + marker_offset) % len(markers)] if use_marker_variation else 'o'
        (point,) = ax.plot(mean_scd, mean_allo, linestyle='None', marker=marker,
                           markersize=marker_size, color=colour,
                           label=None if label_points else label)
        
        if show_errorbars:
            ax.errorbar(mean_scd, mean_allo,xerr=sem_scd, yerr=sem_allo,
                fmt='none', ecolor=point.get_color(), capsize=4, alpha=0.7)
        if label_points:
            ax.text(mean_scd, mean_allo, label, fontsize=9, ha='right', va='bottom')
    
    if not label_points:
        # Right-side legend tuned for print readability.
        legend_ncol = 2 if legend_ncol is None else legend_ncol
        ax.legend(title=r"$\lambda_3$", ncol=legend_ncol,
                  fontsize=legend_fontsize, title_fontsize=legend_title_fontsize,
                  bbox_to_anchor=(1.02, 0.5),
                  loc='center left', frameon=True, columnspacing=1.0, handletextpad=0.5,
                  labelspacing=0.55, borderaxespad=0.0, markerscale=1.25)
    
    # Axis labels and optional title
    ax.set_xlabel("Mean Number of Transhipped Units for SCD Orders", fontsize=fontsize,)
    ax.xaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
    ax.set_ylabel("Mean xA", fontsize=fontsize,)
    ax.set_ylim(bottom=0)
    if title is not None:
        ax.set_title(title, fontsize=14)
    
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout(rect=[0, 0, 0.78, 1])
    
    # Save if requested
    if pdf is not None:
        fig.savefig(pdf, format='pdf', bbox_inches='tight')
        # print(f"✅ Plot saved to {pdf}")
    
    plt.show()
    # return fig, ax


def plot_age_of_units_when_moved(datafilename: str, shu_dict: dict,
                                 figsize=(8,6), dpi=200, pdf=None, title=None,
                            date_moved_col=6, unit_date_bled_col=2, warmup=7*6*3, shelf_life=35,
                            origin_col=3, move_reason=None, move_reason_col=8,):
    if datafilename.endswith('_output.tsv'):
        datafilename = datafilename.replace('_output.tsv', '_moves.npz')
    data = np.load(os.path.realpath(os.path.expanduser(datafilename)))
    all_ages = []
    ages_by_location = {shu_id: [] for shu_id in shu_dict.keys()}
    bins = np.arange(0, shelf_life + 1)
    for replication in data.files:
        moves = data[replication]
        moves = moves[moves[:, date_moved_col] > warmup]
        moves = moves if move_reason is None else moves[moves[:, move_reason_col] == move_reason]
        ages = moves[:, date_moved_col] - moves[:, unit_date_bled_col] + 1
        # all_ages.append(ages)
        rep_counts, _ = np.histogram(ages, bins=bins)
        all_ages.append(rep_counts)
        for shu_id in shu_dict.keys():
            shu_ages = ages[moves[:, origin_col] == shu_id]
            # ages_by_location[shu_id].append(shu_ages)
            shu_rep_counts, _ = np.histogram(shu_ages, bins=bins)
            ages_by_location[shu_id].append(shu_rep_counts)
    # all_ages = np.concatenate(all_ages)
    # counts, bins = np.histogram(all_ages, bins=np.arange(0, shelf_life + 2) - 0.5)
    # counts_avg = counts / len(data.files)
    counts_avg = np.mean(all_ages, axis=0)
    counts_std_err = np.std(all_ages, axis=0) / np.sqrt(len(all_ages))
    shu_counts_avg = {shu_id: np.mean(ages_by_location[shu_id], axis=0) for shu_id in shu_dict.keys()}
    shu_counts_std_err = {shu_id: np.std(ages_by_location[shu_id], axis=0) / np.sqrt(len(ages_by_location[shu_id])) for shu_id in shu_dict.keys()}
    list_ids = list(shu_dict.keys())
    list_ids.insert(len(list_ids)//2, None)
    fig, axs = plt.subplots(5, 3, figsize=figsize, dpi=dpi)
    for i, ax in enumerate(axs.flatten()):
        shu_id = list_ids[i]
        if shu_id is not None:
            ax.stairs(shu_counts_avg[shu_id], bins - 0.5, fill=True, color='gray', alpha=1)
            ax.stairs(shu_counts_avg[shu_id] + shu_counts_std_err[shu_id], bins - 0.5, fill=False, color='black', alpha=0.4, linestyle='--')
            ax.stairs(shu_counts_avg[shu_id] - shu_counts_std_err[shu_id], bins - 0.5, fill=False, color='black', alpha=0.4, linestyle='--')
            ax.set_title(shu_dict[shu_id][shu_dict[shu_id].find('_')+1:])
            ax.set_xticks([0, 6, 12, 18, 24, ])
        else:
            ax.stairs(counts_avg, bins - 0.5, fill=True, color='gray', alpha=1)
            ax.stairs(counts_avg + counts_std_err, bins - 0.5, fill=False, color='black', alpha=0.4, linestyle='--')
            ax.stairs(counts_avg - counts_std_err, bins - 0.5, fill=False, color='black', alpha=0.4, linestyle='--')
            ax.set_title('All Locations')
        ax.set_xticks([0, 6, 12, 18, 24, ])
        ax.set_xlabel('Age of Units when Moved (days)')
        ax.set_ylabel('Avg. frequency')
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if pdf is not None:
        fig.savefig(pdf, bbox_inches='tight')
    fig.show()
    data.close()
    

def print_top_bottom_pairs_unidirectional(dfs: list[pd.DataFrame], n=10, labels=None,
                                          num_unique_shus=14,):
    
    mats_labels = ['SHUs Zero Transportation', 'SHUs Transportation Penalty',]
    if labels is not None:
        mats_labels = labels
    for i, mat in enumerate(dfs):
        # pandas: returns a Series indexed by (row_index, col_name)
        top5 = mat.stack().nlargest(n)
        bottom5 = mat.stack().nsmallest(np.sum(mat.stack().values == 0) + n)  # to avoid zeros
        # as list of (row, col, value)
        top5_list = [(idx[0], idx[1], v) for idx, v in top5.items()]
        bottom5_list = [(idx[0], idx[1], v) for idx, v in bottom5.items() if idx[0] != idx[1]]
        len_bot5 = len(bottom5_list)
        bot5_end = num_unique_shus**2 - 2*num_unique_shus + 1
        bot5_start = bot5_end - len_bot5
        print(f'Top/Bottom {n} unidirectional shipping pairs for {mats_labels[i]}:')
        print(f'Top {n} shipping pairs (source, dest, units moved):')
        print(pd.DataFrame(top5_list, columns=['Source SHU', 'Destination SHU', 'Units Moved'], index=range(1,n+1)))
        # print(top5_list)
        print(f'Bottom {n} shipping pairs (source, dest, units moved):')
        print(pd.DataFrame(bottom5_list[::-1], columns=['Source SHU', 'Destination SHU', 'Units Moved'], index=range(bot5_start, bot5_end)))
        print()


def top_bottom_pairs_bidirectional(df: pd.DataFrame, n=10, *, both=False, nan_as=0.0):
    """
    Return top and/or bottom n unordered pairs (i,j) i!=j ranked by s_ij = df.loc[i,j] + df.loc[j,i].

    Parameters:
    - df: square DataFrame with same index and columns
    - n: number of pairs
    - both: if True returns (top_df, bottom_df); if False returns top_df
    - nan_as: value to substitute for NaN before scoring

    Returns:
    - DataFrame (top n) or tuple(DataFrame top n, DataFrame bottom n) if both=True
    """
    A = df.to_numpy(dtype=float)
    A = np.nan_to_num(A, nan=nan_as)
    S = A + A.T

    i_idx, j_idx = np.triu_indices(S.shape[0], k=1)
    scores = S[i_idx, j_idx]

    top_order = np.argsort(scores)[::-1][:n]
    bottom_order = np.argsort(scores)[:n]

    top_rows = [{'i': df.index[i_idx[k]], 'j': df.columns[j_idx[k]], 'score': float(scores[k])}
                for k in top_order]
    bottom_rows = [{'i': df.index[i_idx[k]], 'j': df.columns[j_idx[k]], 'score': float(scores[k])}
                   for k in bottom_order][::-1]

    top_df = pd.DataFrame(top_rows).reset_index(drop=True)
    bottom_df = pd.DataFrame(bottom_rows).reset_index(drop=True)

    return (top_df, bottom_df) if both else top_df


def dist_moves_per_day_dashboard(files: dict[str, str], shus: dict, warm_up=0, move_reason=3, file_labels=None,
                            origin_col=3, date_moved_col=6, pat_grp_col=8,):
    """
    Loads move data from npz files, filters it, and plots the distribution
    of daily moves.

    By default (separate_plots=False), plots a 5x3 grid:
    - The central plot shows all SHUs combined.
    - The 14 surrounding plots each show an individual SHU.

    If separate_plots=True, generates 15 separate figures (one for each SHU
    and one for 'All SHUs'). If pdf is provided, these are saved
    as a single multi-page PDF.

    Args:
        files (dict): Dictionary of scenario names to file paths.
        shus (dict): Dictionary of SHU IDs to SHU names.
        warm_up (int): Day to start analysis from.
        move_reason (int): Patient group/move reason to filter for.
        file_labels (dict, optional): Alternative labels for scenarios.
        origin_col (int): Column index for origin SHU ID.
        date_moved_col (int): Column index for move date.
        pat_grp_col (int): Column index for patient group/move reason.
        figsize (tuple): Figure size. Default is (20, 25) for grid,
                         recommend (10, 6) for separate plots.
        dpi (int): Figure resolution.
        pdf (str, optional): Path to save the plot. If separate_plots=True,
                                this will be a multi-page PDF.
        plot_type (str): 'violin', 'strip', or 'cdf'.
        jitter (bool): For strip plots.
        dodge (bool): For strip plots.
        legend_title (str): Title for the legend.
        separate_plots (bool): If True, generate separate plots instead of a grid.
    """
    ids = shus.keys()
    names = list(shus.values())
    data_for_df = []
    
    for f_key in files.keys():
        f = files[f_key]
        if f.endswith('_output.tsv'):
            f = f.replace('_output.tsv', '_moves.npz')
        data = np.load(f)
        for replication in data.files:
            moves = data[replication]
            steady_state_moves = moves[moves[:, date_moved_col] > warm_up]
            if type(move_reason) == int:
                steady_state_moves = steady_state_moves[steady_state_moves[:, pat_grp_col] == move_reason]
            for i, id in enumerate(ids):
                shu_name = shus[id]
                shu_moves = steady_state_moves[steady_state_moves[:, origin_col] == id]
                if shu_moves.shape[0] == 0:
                    continue
                unique_days, counts_per_day = np.unique(shu_moves[:, date_moved_col], return_counts=True)
                scenario_name = f_key if file_labels is None else file_labels.get(f_key, f_key)
                data_to_append = [{'daily_moves': count, 'scenario': scenario_name, 'shu': shu_name} for count in counts_per_day]
                data_for_df.extend(data_to_append)
        data.close()
      
    df = pd.DataFrame(data_for_df)
    df['shu'] = df['shu'].str.replace(r'^SHU_', '', regex=True)
    shu_names_list = list(shus.values())
    
    # ---------------------------------------------------------
    # 2. Data Preprocessing
    # ---------------------------------------------------------
    # Create a copy of the data labeled 'ALL' for the SHU column
    df_all = df.copy()
    df_all['shu'] = 'ALL (Total)'

    # Combine original data with the 'ALL' data
    # This allows us to plot specific SHUs and the Total in the same grid
    df_combined = pd.concat([df, df_all], ignore_index=True)

    # Get ranges for sliders
    min_move = df['daily_moves'].min()
    max_move = df['daily_moves'].max()
    
    # --- 2. PLOTTING LOGIC (Internal Closure) ---
    def update_plots(percentile, threshold):
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(14, 14), constrained_layout=True)
        
        # -- Plot 1: Inverse CDF --
        q = percentile / 100.0
        pivot_inv = df_combined.pivot_table(
            index='scenario', columns='shu', values='daily_moves', 
            aggfunc=lambda x: np.quantile(x, q)
        )
        sns.heatmap(pivot_inv, annot=True, fmt=".0f", cmap="viridis", ax=axes[0],
                    vmin=min_move, vmax=max_move, 
                    cbar_kws={'label': 'Daily RBC Transfers'})
        # axes[0].set_title(f'Inverse CDF: Daily Moves Value at {percentile}% Percentile', fontsize=14)
        axes[0].set_title(f'{percentile:.2f}% of transfers in one day are below the value(s) shown', fontsize=22, pad=12)
        axes[0].set_xlabel('')
        axes[0].set_ylabel('Transportation Penalty Scenario: $\lambda_3$')
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')


        # -- Plot 2: Complementary CDF --
        pivot_ccdf = df_combined.pivot_table(
            index='scenario', columns='shu', values='daily_moves', 
            aggfunc=lambda x: (x > threshold).mean()
        )
        sns.heatmap(pivot_ccdf, annot=True, fmt=".0%", cmap="magma", ax=axes[1], 
                    vmin=0, vmax=1, cbar_kws={'label': 'Probability'})
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')
        axes[1].set_title(f'Probability that a day\'s transfers exceeds {threshold:.0f} RBC(s)', fontsize=22, pad=12)
        axes[1].set_xlabel('SHU Location')
        axes[1].set_ylabel('Transportation Penalty Scenario: $\lambda_3$')        
        
        plt.show()

    # --- 3. WIDGET CREATION ---
    style = {'description_width': 'initial'}
    
    pct_slider = widgets.FloatSlider(
        value=95.0, min=0.0, max=100.0, step=0.05, 
    description='Show transfer values for this percentile of days: (%)', style=style, layout=widgets.Layout(width='90%')
    )
    
    val_slider = widgets.IntSlider(
        value=50, min=min_move, max=max_move, step=1, 
        description='Threshold for Number of Units to Transfer Out:', style=style, layout=widgets.Layout(width='90%')
    )

    # --- 4. OUTPUT & LAYOUT ---
    # Using interactive_output allows the logic to stay inside this function
    # while capturing the local variable 'df_combined'
    out_display = widgets.interactive_output(update_plots, {'percentile': pct_slider, 'threshold': val_slider})
    
    # Assemble the dashboard
    dashboard = widgets.VBox([
        widgets.HTML("<h1>Number of Daily RBC Transfers by Transportation Scenario</h1>"),
        pct_slider, 
        val_slider, 
        widgets.HTML("<hr>"),
        out_display
    ])
    
    return dashboard
    
    
def moves_per_day_df(files: dict[str, str], shus: dict, warm_up=0, time_horizon=7*6*1, move_reason=3, file_labels=None,
                            origin_col=3, date_moved_col=6, pat_grp_col=8,
                            out_file=None, keep_zero_days=False,):
    """
    Loads move data from npz files, filters it, and prepares a DataFrame
    of the distribution of daily moves.
    
    Args:
        files (dict): Dictionary of scenario names to file paths.
        shus (dict): Dictionary of SHU IDs to SHU names.
        warm_up (int): Day to start analysis from.
        move_reason (int): Patient group/move reason to filter for.
        file_labels (dict, optional): Alternative labels for scenarios.
        origin_col (int): Column index for origin SHU ID.
        date_moved_col (int): Column index for move date.
        pat_grp_col (int): Column index for patient group/move reason.
    """
    ids = shus.keys()
    names = list(shus.values())
    data_for_df = []
    
    for f_key in files.keys():
        f = files[f_key]
        if f.endswith('_output.tsv'):
            f = f.replace('_output.tsv', '_moves.npz')
        data = np.load(f)
        for replication in data.files:
            moves = data[replication]
            steady_state_moves = moves[moves[:, date_moved_col] > warm_up]
            if type(move_reason) == int:
                steady_state_moves = steady_state_moves[steady_state_moves[:, pat_grp_col] == move_reason]
            for i, id in enumerate(ids):
                shu_name = shus[id]
                shu_moves = steady_state_moves[steady_state_moves[:, origin_col] == id]
                if shu_moves.shape[0] == 0 and not keep_zero_days:
                    continue
                unique_days, counts_per_day = np.unique(shu_moves[:, date_moved_col], return_counts=True)
                counts_per_day_full = np.hstack((counts_per_day, np.zeros(time_horizon - warm_up - len(counts_per_day))))
                counts_per_day_full = counts_per_day_full.astype(int) if keep_zero_days else counts_per_day
                scenario_name = f_key if file_labels is None else file_labels.get(f_key, f_key)
                data_to_append = [{'daily_moves': count, 'scenario': scenario_name, 'shu': shu_name} for count in counts_per_day_full]
                data_for_df.extend(data_to_append)
        data.close()
      
    df = pd.DataFrame(data_for_df)
    if out_file is not None:
        df.to_csv(out_file, index=False, sep='\t')
    return df

