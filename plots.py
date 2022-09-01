import json
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import math
import json
from scipy.stats import ttest_ind
from utils import get_stat_func, CompressPDF

font = {'size': 70}
matplotlib.rc('font', **font)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams['figure.constrained_layout.use'] = True

colors = [[0, 0.447, 0.7410], [0.85, 0.325, 0.098],  [0.466, 0.674, 0.188], [0.929, 0.694, 0.125],
          [0.494, 0.1844, 0.556],[0.3010, 0.745, 0.933], [137/255,145/255,145/255],
          [0.466, 0.674, 0.8], [0.929, 0.04, 0.125],
          [0.3010, 0.245, 0.33], [0.635, 0.078, 0.184], [0.35, 0.78, 0.504]]
cmap = plt.get_cmap('tab10')
# colors = np.array(cmap.colors)[[0, 17, 2, 14, 5, 8, 9, 12, 13, 14, 16, 17, 18]]
colors = np.array(cmap.colors)
folder = 'query_study'

RESULTS_PATH = '/home/ahmed/Documents/Amaterasu/hachibi/active_hme/results/iclr2023/' + folder + '/'
SAVE_PATH = '/home/ahmed/Documents/Amaterasu/hachibi/active_hme/plots/iclr2023/'
# TO_PLOT = ['_global_sr']
TO_PLOT = ['proposed_beyond']

metric_to_label = {'stepping_stones_len': '# Stepping Stones', 'query_proba': 'Query probability', 'query_proba_intern': 'Internalization probability',
                   'nb_discovered': '# Discovered goals', 'proposed_beyond': '# Proposed beyond', 'proposed_ss': '# Proposed SS', 
                   'nb_internalized_pairs': '# Internalized goals', 'agent_nodes': '# Agent nodes'}
NB_CLASSES = 11 # 12 for 5 blocks

LINE = 'mean'
ERR = 'std'
DPI = 30
N_SEEDS = None
N_EPOCHS = None
LINEWIDTH = 8 # 8 for per class
MARKERSIZE = 15 # 15 for per class
ALPHA = 0.3
ALPHA_TEST = 0.05
MARKERS = ['o', 'v', 's', 'P', 'D', 'X', "*", 'v', 's', 'p', 'P', '1']
FREQ = 1
NB_BUCKETS = 10
NB_EPS_PER_EPOCH = 2000
NB_VALID_GOALS = 35
LAST_EP = 122
LIM = NB_EPS_PER_EPOCH * LAST_EP / 1000 + 5
line, err_min, err_plus = get_stat_func(line=LINE, err=ERR)
COMPRESSOR = CompressPDF(4)
# 0: '/default',
# 1: '/prepress',
# 2: '/printer',
# 3: '/ebook',
# 4: '/screen'


def setup_figure(xlabel=None, ylabel=None, xlim=None, ylim=None):
    fig = plt.figure(figsize=(37, 20), frameon=False) # 34 18 for semantic
    ax = fig.add_subplot(111)
    ax.spines['top'].set_linewidth(6)
    ax.spines['right'].set_linewidth(6)
    ax.spines['bottom'].set_linewidth(6)
    ax.spines['left'].set_linewidth(6)
    ax.tick_params(width=10, direction='in', length=20, labelsize='60')
    artists = ()
    if xlabel:
        xlab = plt.xlabel(xlabel)
        artists += (xlab,)
    if ylabel:
        ylab = plt.ylabel(ylabel)
        artists += (ylab,)
    if ylim:
        plt.ylim(ylim)
    if xlim:
        plt.xlim(xlim)
    return artists, ax

def setup_n_figs(n, m, xlabels=None, ylabels=None, xlims=None, ylims=None):
    # if n == 1:
    #     fig, axs = plt.subplots(n, m, figsize=(64, 12), frameon=False)
    # else:
    #     fig, axs = plt.subplots(n, m, figsize=(m * 18, n * 12), frameon=False)
    # axs = axs.ravel()
    # artists = ()
    # for i_ax, ax in enumerate(axs):
    #     ax.spines['top'].set_linewidth(3)
    #     ax.spines['right'].set_linewidth(3)
    #     ax.spines['bottom'].set_linewidth(3)
    #     ax.spines['left'].set_linewidth(3)
    #     ax.tick_params(width=7, direction='in', length=15, labelsize='55', zorder=10)
    #     if xlabels[i_ax]:
    #         xlab = ax.set_xlabel(xlabels[i_ax])
    #         artists += (xlab,)
    #     if ylabels[i_ax]:
    #         ylab = ax.set_ylabel(ylabels[i_ax])
    #         artists += (ylab,)
    #     if ylims[i_ax]:
    #         ax.set_ylim(ylims[i_ax])
    #     if xlims[i_ax]:
    #         ax.set_xlim(xlims[i_ax])
    # fig, axs = plt.subplots(2, 3, figsize=(25, 10), frameon=False)
    fig, axs = plt.subplots(n, m, figsize=(12*m,10*n), frameon=False)
    axs = axs.ravel()
    # fig = plt.figure(figsize=(22, 7))
    # axs = np.array([plt.subplot(141, gridspec_kw={'width_ratios': [3, 1]}), plt.subplot(142, gridspec_kw={'width_ratios': [3, 1]}),
    #                 plt.subplot(143, gridspec_kw={'width_ratios': [3, 1]}), plt.subplot(144, gridspec_kw={'width_ratios': [3, 1]})])
    axs = axs.ravel()
    artists = ()
    for i_ax, ax in enumerate(axs):
        ax.spines['top'].set_linewidth(3)
        ax.spines['right'].set_linewidth(3)
        ax.spines['bottom'].set_linewidth(3)
        ax.spines['left'].set_linewidth(3)
        ax.tick_params(width=5, direction='in', length=15, labelsize='50', zorder=10)
        if xlabels[i_ax]:
            xlab = ax.set_xlabel(xlabels[i_ax])
            artists += (xlab,)
        if ylabels[i_ax]:
            ylab = ax.set_ylabel(ylabels[i_ax])
            artists += (ylab,)
        if ylims[i_ax]:
            ax.set_ylim(ylims[i_ax])
        if xlims[i_ax]:
            ax.set_xlim(xlims[i_ax])
    return fig, artists, axs

def save_fig(path, artists):
    plt.savefig(os.path.join(path), bbox_extra_artists=artists, bbox_inches='tight', dpi=DPI)
    plt.close('all')
    # compress PDF
    try:
        COMPRESSOR.compress(path, path[:-4] + '_compressed.pdf')
        os.remove(path)
    except:
        pass


def check_length_and_seeds(experiment_path):
    conditions = os.listdir(experiment_path)
    # check max_length and nb seeds
    max_len = 0
    max_seeds = 0
    min_len = 1e6
    min_seeds = 1e6

    for cond in conditions:
        cond_path = experiment_path + cond + '/'
        list_runs = sorted(os.listdir(cond_path))
        if len(list_runs) > max_seeds:
            max_seeds = len(list_runs)
        if len(list_runs) < min_seeds:
            min_seeds = len(list_runs)
        for run in list_runs:
            try:
                run_path = cond_path + run + '/'
                data_run = pd.read_csv(run_path + 'progress.csv')
                nb_epochs = len(data_run)
                if nb_epochs > max_len:
                    max_len = nb_epochs
                if nb_epochs < min_len:
                    min_len = nb_epochs
            except:
                pass
    return max_len, max_seeds, min_len, min_seeds

def get_stats_from_label(experiment_path, to_plot, max_seeds, conditions):
    probas = np.zeros([max_seeds, len(conditions), LAST_EP + 1 ])
    probas.fill(np.nan)
    for i_cond, cond in enumerate(conditions):
        cond_path = experiment_path + cond + '/'
        list_runs = sorted(os.listdir(cond_path))
        for i_run, run in enumerate(list_runs):
            run_path = cond_path + run + '/'
            data_run = pd.read_csv(run_path + 'progress.csv')
            p= np.array(data_run[to_plot][:LAST_EP + 1])
            probas[i_run, i_cond, :p.size] = p.copy()


    probas_per_cond_stats = np.zeros([len(conditions), LAST_EP + 1, 3])
    probas_per_cond_stats[:, :, 0] = line(probas)
    probas_per_cond_stats[:, :, 1] = err_min(probas)
    probas_per_cond_stats[:, :, 2] = err_plus(probas)

    return probas_per_cond_stats

def plot_sr_av(max_len, experiment_path, folder):

    condition_path = experiment_path + folder + '/'
    list_runs = sorted(os.listdir(condition_path))
    global_sr = np.zeros([len(list_runs), max_len])
    global_sr.fill(np.nan)
    sr_data = np.zeros([len(list_runs), NB_CLASSES, max_len])
    sr_data.fill(np.nan)
    x_eps = np.arange(0, (LAST_EP + 1) * NB_EPS_PER_EPOCH, NB_EPS_PER_EPOCH * FREQ) / 1000
    # x_eps = np.arange(0, max_len, FREQ)
    x = np.arange(0, LAST_EP + 1, FREQ)
    for i_run, run in enumerate(list_runs):
        run_path = condition_path + run + '/'
        data_run = pd.read_csv(run_path + 'progress.csv')

        T = len(data_run['Eval_SR_1'][:LAST_EP + 1])
        SR = np.zeros([NB_CLASSES, T])
        for t in range(T):
            for i in range(NB_CLASSES):
                SR[i, t] = data_run['Eval_SR_{}'.format(i+1)][t]
        all_sr = np.mean([data_run['Eval_SR_{}'.format(i+1)] for i in range(NB_CLASSES)], axis=0)

        sr_buckets = []
        for i in range(SR.shape[0]):
            sr_buckets.append(SR[i])
        sr_buckets = np.array(sr_buckets)
        sr_data[i_run, :, :sr_buckets.shape[1]] = sr_buckets.copy()
        global_sr[i_run, :all_sr.size] = all_sr.copy()

    artists, ax = setup_figure(  # xlabel='Episodes (x$10^3$)',
        xlabel='Episodes (x$10^3$)',
        ylabel='Success Rate',
        xlim=[-1, LIM],
        ylim=[-0.02, 1.03])
    sr_per_cond_stats = np.zeros([NB_CLASSES, max_len, 3])
    sr_per_cond_stats[:, :, 0] = line(sr_data)
    sr_per_cond_stats[:, :, 1] = err_min(sr_data)
    sr_per_cond_stats[:, :, 2] = err_plus(sr_data)
    av = line(global_sr)
    for i in range(NB_CLASSES):
        plt.plot(x_eps, sr_per_cond_stats[i, x, 0], color=colors[i], marker=MARKERS[i], markersize=MARKERSIZE, linewidth=LINEWIDTH)
    plt.plot(x_eps, av[x], color=[0.3]*3, linestyle='--', linewidth=LINEWIDTH // 2)
    leg = plt.legend(['Class {}'.format(i+1) for i in range(NB_CLASSES)] + ['Global'],
                     loc='upper center',
                     bbox_to_anchor=(0.5, 1.15),
                     ncol=6,
                     fancybox=True,
                     shadow=True,
                     prop={'size': 30, 'weight': 'bold'},
                     markerscale=1)
    artists += (leg,)

    for i in range(NB_CLASSES):
        plt.fill_between(x_eps, sr_per_cond_stats[i, x, 1], sr_per_cond_stats[i, x, 2], color=colors[i], alpha=ALPHA)

    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    plt.grid()
    # ax.set_facecolor((244/255, 244/255, 244/255))
    save_fig(path=SAVE_PATH + folder + '_sr.pdf', artists=artists)


def plot_sr_av_all(max_len, experiment_path, titles, folders):
    m = len(folders)
    fig, artists, ax = setup_n_figs(n=1,
                                   m=m, 
                                #    xlabels=[None, None, 'Episodes (x$10^3$)', 'Episodes (x$10^3$)'],
                                #    ylabels= ['Success Rate', None] * 2,
                                   xlabels = ['Episodes (x$10^3$)'] * m,
                                   ylabels = ['Success Rate'] + [None] * (m-1),
                                   xlims = [[-1, LIM] for _ in range(m)],
                                   ylims= [[-0.02, 1.03] for _ in range(m)]
        )
    for k, folder in enumerate(folders):
        condition_path = experiment_path + folder + '/'
        list_runs = sorted(os.listdir(condition_path))
        global_sr = np.zeros([len(list_runs), max_len])
        global_sr.fill(np.nan)
        sr_data = np.zeros([len(list_runs), NB_CLASSES, max_len])
        sr_data.fill(np.nan)
        x_eps = np.arange(0, (LAST_EP + 1) * NB_EPS_PER_EPOCH, NB_EPS_PER_EPOCH * FREQ) / 1000
        # x_eps = np.arange(0, max_len, FREQ)
        x = np.arange(0, LAST_EP + 1, FREQ)
        for i_run, run in enumerate(list_runs):
            run_path = condition_path + run + '/'
            data_run = pd.read_csv(run_path + 'progress.csv')

            T = len(data_run['Eval_SR_1'][:LAST_EP + 1])
            SR = np.zeros([NB_CLASSES, T])
            for t in range(T):
                for i, j in enumerate([0, 1, 3, 4, 5, 6, 7, 8, 9, 10]):
                    SR[i, t] = data_run['Eval_SR_{}'.format(j+1)][t]
            all_sr = np.mean([data_run['Eval_SR_{}'.format(i+1)] for i in [0, 1, 3, 4, 5, 6, 7, 8, 9, 10]], axis=0)

            sr_buckets = []
            for i in range(SR.shape[0]):
                sr_buckets.append(SR[i])
            sr_buckets = np.array(sr_buckets)
            sr_data[i_run, :, :sr_buckets.shape[1]] = sr_buckets.copy()
            global_sr[i_run, :all_sr.size] = all_sr.copy()
        
        sr_per_cond_stats = np.zeros([NB_CLASSES, max_len, 3])
        sr_per_cond_stats[:, :, 0] = line(sr_data)
        sr_per_cond_stats[:, :, 1] = err_min(sr_data)
        sr_per_cond_stats[:, :, 2] = err_plus(sr_data)
        av = line(global_sr)
        for i in range(NB_CLASSES):
            ax[k].plot(x_eps, sr_per_cond_stats[i, x, 0], color=colors[i], marker=MARKERS[i], markersize=MARKERSIZE, linewidth=LINEWIDTH)
        # ax[k].plot(x_eps, av[x], color=[0.3]*3, linestyle='--', linewidth=LINEWIDTH // 2)

        for i in range(NB_CLASSES):
            ax[k].fill_between(x_eps, sr_per_cond_stats[i, x, 1], sr_per_cond_stats[i, x, 2], color=colors[i], alpha=ALPHA)

        ax[k].set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax[k].grid()
        ax[k].set_title(titles[k], fontname='monospace', fontweight='bold')
        ax[k].set_facecolor('whitesmoke')
    leg = fig.legend(['$C_1$', '$C_2$', '$S_2$', '$S_3$', '$S_2$ & $S_2$', '$S_2$ & $S_3$', '$P_3$', '$P_3$ & $S_2$', '$S_4$', '$S_5$'],
                    #['No Stacks', '$\widetilde{S}_2$', '$\widetilde{S}_3$', '$\widetilde{S}_4$', '$\widetilde{S}_5$', 'Global'],
                    loc='upper center',
                    bbox_to_anchor=(0.505, 1.22),
                    ncol=10,
                    fancybox=True,
                    shadow=True,
                    prop={'size': 75, 'weight': 'normal'},
                    markerscale=1)
    artists += (leg,)
    save_fig(path=SAVE_PATH + 'per_class.pdf', artists=artists)


def get_mean_sr(experiment_path, max_len, max_seeds, conditions=None, labels=None, ref='with_init'):
    if conditions is None:
        conditions = os.listdir(experiment_path)
    sr = np.zeros([max_seeds, len(conditions), LAST_EP + 1 ])
    sr.fill(np.nan)
    for i_cond, cond in enumerate(conditions):
        if cond == ref:
            ref_id = i_cond
        cond_path = experiment_path + cond + '/'
        list_runs = sorted(os.listdir(cond_path))
        for i_run, run in enumerate(list_runs):
            run_path = cond_path + run + '/'
            data_run = pd.read_csv(run_path + 'progress.csv')
            all_sr = np.mean(np.array([data_run['Eval_SR_{}'.format(i+1)][:LAST_EP + 1] for i in range(NB_CLASSES) if i!=2]), axis=0)
            sr[i_run, i_cond, :all_sr.size] = all_sr.copy()


    sr_per_cond_stats = np.zeros([len(conditions), LAST_EP + 1, 3])
    sr_per_cond_stats[:, :, 0] = line(sr)
    sr_per_cond_stats[:, :, 1] = err_min(sr)
    sr_per_cond_stats[:, :, 2] = err_plus(sr)


    x_eps = np.arange(0, (LAST_EP + 1) * NB_EPS_PER_EPOCH, NB_EPS_PER_EPOCH * FREQ) / 1000
    x = np.arange(0, LAST_EP + 1, FREQ)
    artists, ax = setup_figure(xlabel='Episodes (x$10^3$)',
                               # xlabel='Epochs',
                               ylabel='Success Rate',
                               xlim=[-1, LIM],
                               ylim=[-0.02, 1 + 0.045 * len(labels)])

    for i in range(len(conditions)):
        plt.plot(x_eps, sr_per_cond_stats[i, x, 0], color=colors[i], marker=MARKERS[i], markersize=MARKERSIZE, linewidth=LINEWIDTH)

    if labels is None:
        labels = conditions
    leg = plt.legend(labels,
                     loc='upper center',
                     bbox_to_anchor=(0.5, 1.15),
                     ncol=7,
                     fancybox=True,
                     shadow=True,
                     prop={'size': 60, 'weight': 'bold'},
                     markerscale=1,
                     )
    for l in leg.get_lines():
        l.set_linewidth(7.0)
    artists += (leg,)
    for i in range(len(conditions)):
        plt.fill_between(x_eps, sr_per_cond_stats[i, x, 1], sr_per_cond_stats[i, x, 2], color=colors[i], alpha=ALPHA)
    
    # compute p value wrt ref id
    p_vals = dict()
    for i_cond in range(len(conditions)):
        if i_cond != ref_id:
            p_vals[i_cond] = []
            for i in x:
                ref_inds = np.argwhere(~np.isnan(sr[:, ref_id, i])).flatten()
                other_inds = np.argwhere(~np.isnan(sr[:, i_cond, i])).flatten()
                if ref_inds.size > 1 and other_inds.size > 1:
                    ref = sr[:, ref_id, i][ref_inds]
                    other = sr[:, i_cond, i][other_inds]
                    p_vals[i_cond].append(ttest_ind(ref, other, equal_var=False)[1])
                else:
                    p_vals[i_cond].append(1)

    i = 0    
    for i_cond in range(len(conditions)):
        if i_cond != ref_id:
            inds_sign = np.argwhere(np.array(p_vals[i_cond]) < ALPHA_TEST).flatten()
            if inds_sign.size > 0:
                plt.scatter(x=x_eps[inds_sign], y=np.ones([inds_sign.size]) + 0.04 + 0.05 * i, marker='*', color=colors[i_cond], s=1000)
            i += 1

    ax.hlines(y=0.93, xmin=0, xmax=(LAST_EP + 1) * NB_EPS_PER_EPOCH / 1000, color='black', linewidth=3, linestyles='dashed')
    ax.set_yticks([0, 0.25, 0.5, 0.75, 0.93, 1])
    plt.grid()
    # ax.set_facecolor((244/255, 244/255, 244/255))
    save_fig(path=SAVE_PATH + folder + PLOT + '.pdf', artists=artists)
    return sr_per_cond_stats.copy()

def get_query_proba(experiment_path, max_seeds, conditions=None, labels=None, to_plot='query_proba', max_value=1.):
    if conditions is None:
        conditions = os.listdir(experiment_path)
    probas = np.zeros([max_seeds, len(conditions), LAST_EP + 1 ])
    probas.fill(np.nan)
    for i_cond, cond in enumerate(conditions):
        cond_path = experiment_path + cond + '/'
        list_runs = sorted(os.listdir(cond_path))
        for i_run, run in enumerate(list_runs):
            run_path = cond_path + run + '/'
            data_run = pd.read_csv(run_path + 'progress.csv')
            p= np.array(data_run[to_plot][:LAST_EP + 1])
            probas[i_run, i_cond, :p.size] = p.copy()


    probas_per_cond_stats = np.zeros([len(conditions), LAST_EP + 1, 3])
    probas_per_cond_stats[:, :, 0] = line(probas)
    probas_per_cond_stats[:, :, 1] = err_min(probas)
    probas_per_cond_stats[:, :, 2] = err_plus(probas)


    x_eps = np.arange(0, (LAST_EP + 1) * NB_EPS_PER_EPOCH, NB_EPS_PER_EPOCH * FREQ) / 1000
    x = np.arange(0, LAST_EP + 1, FREQ)
    artists, ax = setup_figure(xlabel='Episodes (x$10^3$)',
                               # xlabel='Epochs',
                               ylabel=metric_to_label[to_plot],
                               xlim=[-1, LIM],
                               ylim=[-0.02, max_value + 0.02])

    for i in range(len(conditions)):
        print(labels[i])
        if to_plot == 'proposed_ss':
            print(f'{probas_per_cond_stats[i, -1, 0]}+={(probas_per_cond_stats[i, -1, 2] - probas_per_cond_stats[i, -1, 0])}')
        else: 
            print(f'{probas_per_cond_stats[i, -1, 0]}+={(probas_per_cond_stats[i, -1, 2] - probas_per_cond_stats[i, -1, 0])}')
        print('===========')
        plt.plot(x_eps, probas_per_cond_stats[i, x, 0], color=colors[i], marker=MARKERS[i], markersize=MARKERSIZE, linewidth=LINEWIDTH)

    if labels is None:
        labels = conditions
    leg = plt.legend(labels,
                     loc='upper center',
                     bbox_to_anchor=(0.5, 1.15),
                     ncol=7,
                     fancybox=True,
                     shadow=True,
                     prop={'size': 60, 'weight': 'bold'},
                     markerscale=1,
                     )
    for l in leg.get_lines():
        l.set_linewidth(7.0)
    artists += (leg,)
    for i in range(len(conditions)):
        plt.fill_between(x_eps, probas_per_cond_stats[i, x, 1], probas_per_cond_stats[i, x, 2], color=colors[i], alpha=ALPHA)
    plt.grid()
    # ax.set_facecolor((244/255, 244/255, 244/255))
    save_fig(path=SAVE_PATH + PLOT + '.pdf', artists=artists)
    return probas_per_cond_stats.copy()

def plot_counts_and_sr(experiment_path, conditions):
    titles = ['Stack 3', 'Stack 4', 'Stack 5']
    classes = [5, 10, 11]
    fig, artists, axs = setup_n_figs(n=1,
                                     m=3,
                                     xlabels=['Episodes (x$10^3$)', 'Episodes (x$10^3$)', 'Episodes (x$10^3$)', 'Episodes (x$10^3$)'],
                                     ylabels=['Goal Counts', None, None, None] ,
                                     xlims=[(0, LIM)] * 4,
                                     ylims=[(0, 100), (0, 100), (0, 100), (0, 100)])
                                     
    for k, cond in enumerate(conditions):
        x_eps = np.arange(0, (LAST_EP + 1) * NB_EPS_PER_EPOCH, NB_EPS_PER_EPOCH * FREQ) / 1000
        x = np.arange(0, LAST_EP + 1, FREQ)
        for i, c in enumerate(classes):
            if i !=0:
                axs[4*k + i].set_yticklabels([])
            axs[4*k + i].set_xticks([0, 100, 200, 300])
            
            ax2 = axs[4*k + i].twinx()
            sr_class =  get_stats_from_label(experiment_path, f'Eval_SR_{c}', max_seeds=3, conditions=[cond])
            l3 = ax2.plot(x_eps, sr_class[0, x, 0], color=colors[i], marker=MARKERS[c], markersize=MARKERSIZE, linewidth=LINEWIDTH,
                        linestyle='solid')
            l3f = ax2.fill_between(x_eps, sr_class[0, x, 1], sr_class[0, x, 2], color=colors[i], alpha=ALPHA)
            if i == len(classes) - 1:
                ax2.tick_params(width=5, direction='in', length=20, labelsize='50')
                ax2.set_ylabel("SR")
                ax2.set_ylim(-0.01, 1.01)
            else:
                ax2.set_yticklabels([])
            
            teacher_goals =  get_stats_from_label(experiment_path, f'# class_teacher {c}', max_seeds=3, conditions=[cond])
            discovered_goals =  get_stats_from_label(experiment_path, f'# class_agent {c}', max_seeds=3, conditions=[cond])
            l1 = axs[4*k + i].plot(x_eps, teacher_goals[0, x, 0], color=colors[i], marker=MARKERS[c], markersize=MARKERSIZE, linewidth=LINEWIDTH,
                        linestyle='dashed')
            l2= axs[4*k + i].plot(x_eps, discovered_goals[0, x, 0], color=colors[i], marker=MARKERS[c], markersize=MARKERSIZE, linewidth=LINEWIDTH,
                        linestyle='dotted')
        
            
            axs[i].set_title(titles[i])
            axs[i].grid()
            axs[i].set_facecolor('whitesmoke')

        # l = l1 + l2 + l3
            if k == 0 and i ==0:
                leg = fig.legend(['# SP suggestions', '# reached configurations', 'Success Rate'],
                                 loc='upper center',
                                 bbox_to_anchor=(0.5, 1.22),
                                 ncol=3,
                                 fancybox=True,
                                 shadow=True,
                                 prop={'size': 60, 'weight': 'bold'},
                                 markerscale=1)
                artists += (leg,)
        # ax.grid()
        # plt.show()
        # save_fig(path=run_path + 'goals_{}.pdf'.format(cond), artists=artists)
        # except:
        #     print('failed')
    plt.savefig(SAVE_PATH + '/goals_sr.pdf'.format(i), bbox_inches='tight')
    plt.close('all')

if __name__ == '__main__':

    for PLOT in TO_PLOT:
        print('\n\tPlotting', PLOT)
        experiment_path = RESULTS_PATH

        max_len, max_seeds, min_len, min_seeds = check_length_and_seeds(experiment_path=experiment_path)

        # conditions = [f'exp_queries_q={q}_beta={b}' for q in [200] for b in [0, 20, 50, 100, 200, 300, 1000]]
        # labels = [f'β={b}' for q in [200] for b in [0, 20, 50, 100, 200, 300, 1000]]
        # get_query_proba(experiment_path, max_seeds, conditions, labels, to_plot=PLOT, max_value=100000.)

        # conditions = [f'internalization_strategy={s}' for s in [4, 2, 3, 1, 0]]
        # labels = [f'IN-{s}' for s in [1, 2, 3, 4]] + ['w/o IN']
        # get_mean_sr(experiment_path, max_len, max_seeds, conditions, labels, ref=conditions[0])
        # plot_sr_av(max_len, experiment_path, 'flat')
        # plot_sr_av_all(max_len, experiment_path, titles=labels, folders=conditions)
        # get_query_proba(experiment_path, max_seeds, conditions, labels, to_plot=PLOT, max_value=50000)

        # Study fixed queries
        # conditions = ['main_agent'] + [f'fixed_proba={p}' for p in [0.05, 0.075, 0.1]]
        # labels = ['Active queries'] + [f'q={p}' for p in [0.05, 0.075, 0.1]]
        # get_mean_sr(experiment_path, max_len, max_seeds, conditions, labels, ref=conditions[0])
        # get_query_proba(experiment_path, max_seeds, conditions, labels, to_plot=PLOT, max_value=50000)

        # Study learning trajectories
        # plot_counts_and_sr(experiment_path=experiment_path, conditions=['main_beta=50'])

        # Study main
        # conditions = [f'main_beta={p}' for p in [0, 20, 50, 100, 200, 500]]
        # labels = [f'β={b}' for b in [0, 20, 50, 100, 200, 500]]
        # Exploration Study
        # conditions = [f'agent={p}' for p in ['HME', 'UniformandRandom', 'F2andRandom', 'F3andRandom']]
        # labels = [f'{b}' for b in ['HME-β=50', 'Uniform&Random', 'F2&Random', 'F3&Random']]
        # ACL Study 
        # conditions = [f'agent={p}' for p in ['HME', 'LPAgent']]
        # labels = [f'{b}' for b in ['HME-β=50', 'LP Baseline']]
        # Fixed queries study
        conditions = [f'agent={p}' for p in ['HME', 'f0.05', 'f0.06', 'f0.07']]
        labels = [f'{b}' for b in ['HME-β=50', 'Fixed 0.05', 'Fixed 0.06', 'Fixed 0.07']]
        # get_mean_sr(experiment_path, max_len, max_seeds, conditions, labels, ref=conditions[0])
        get_query_proba(experiment_path, max_seeds, conditions, labels, to_plot=PLOT, max_value=150)