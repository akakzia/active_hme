import torch
import os
from turtle import title
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle as pkl
from scipy.stats import ttest_ind
from utils import get_stat_func, CompressPDF, get_env_params
from rl_modules.rn_models import RnSemantic
from arguments import get_args

font = {'size': 80}
matplotlib.rc('font', **font)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams['figure.constrained_layout.use'] = True

colors = [[0, 0.447, 0.7410], [0.85, 0.325, 0.098],  [0.466, 0.674, 0.188], [0.929, 0.694, 0.125],
          [0.494, 0.1844, 0.556],[0.3010, 0.745, 0.933], [137/255,145/255,145/255],
          [0.466, 0.674, 0.8], [0.929, 0.04, 0.125],
          [0.3010, 0.245, 0.33], [0.635, 0.078, 0.184], [0.35, 0.78, 0.504]]
cmap = plt.get_cmap('tab10')
# colors = np.array(cmap.colors)[[14, 17, 2, 0, 5, 8, 9, 12, 13, 14, 16, 17, 18]]
colors = np.array(cmap.colors)
folder = 'lp_estimation'

RESULTS_PATH = '/home/ahmed/Documents/Amaterasu/hachibi/active_hme/results/iclr2023/' + folder + '/'
SAVE_PATH = '/home/ahmed/Documents/Amaterasu/hachibi/active_hme/plots/iclr2023/'

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
FREQ = 5
NB_BUCKETS = 10
NB_EPS_PER_EPOCH = 2000
NB_VALID_GOALS = 35
LAST_EP = 130
LIM = NB_EPS_PER_EPOCH * LAST_EP / 1000 + 5
line, err_min, err_plus = get_stat_func(line=LINE, err=ERR)
COMPRESSOR = CompressPDF(4)
# 0: '/default',
# 1: '/prepress',
# 2: '/printer',
# 3: '/ebook',
# 4: '/screen'


def setup_figure(xlabel=None, ylabel=None, xlim=None, ylim=None):
    fig = plt.figure(figsize=(42, 25), frameon=False) # 34 18 for semantic
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


def get_mean_sr(experiment_path, max_len, max_seeds, conditions=None, labels=None, ref='with_init'):
    if conditions is None:
        conditions = os.listdir(experiment_path)
    sr = np.zeros([max_seeds, len(conditions), LAST_EP + 1 ])
    sr.fill(np.nan)
    all_goal_values = []
    for i_cond, cond in enumerate(conditions):
        if cond == ref:
            ref_id = i_cond
        cond_path = experiment_path + cond + '/'
        list_runs = sorted(os.listdir(cond_path))
        for i_run, run in enumerate(list_runs):
            run_path = cond_path + run + '/'
            models_path = cond_path + run + '/models'
            goals_path = cond_path + run + '/buckets'
            values_per_run = []
            for ep in range(0, LAST_EP + 1, 10):
                running_model = models_path + f'/model_{ep}.pt'
                running_goals = goals_path + f'/discovered_g_ep_{LAST_EP}.pkl'
                args = get_args()
                # Make the environment
                env_params = {'obs': 85, 'goal': 35, 'action': 4, 'action_max': 1, 'max_timesteps': 200}
                model = RnSemantic(env_params, args)
                _, _, _, _, value_network = torch.load(running_model, map_location=lambda storage, loc: storage)
                model.value_network.load_state_dict(value_network)
                with open(running_goals, 'rb') as f:
                    agent_goals = pkl.load(file=f)
                g_tensor = torch.tensor(agent_goals, dtype=torch.float32)
                ag = - np.ones(np.array(agent_goals).shape).astype(np.float)
                ag_tensor = torch.tensor(ag, dtype=torch.float32)
                with torch.no_grad():
                    values = model.value_network.forward(ag_tensor, g_tensor).numpy()
                max_value = 5
                min_value = 0
                norm_goals = np.clip((values - min_value)/(max_value - min_value), a_min=0., a_max=1.)
                values_per_run.append(norm_goals)
            all_goal_values.append(np.array(values_per_run))
            data_run = pd.read_csv(run_path + 'progress.csv')
            p= np.array(data_run['nb_discovered'][:LAST_EP + 1])
            p = np.array([p[j] for j in range(0, LAST_EP + 1, 10)])
            lp_goals = np.array([abs(values_per_run[j+1] - values_per_run[j]) for j in range(len(values_per_run) - 1)])
            lp_goals = np.squeeze(lp_goals)
            global_lp = [np.mean(lp_goals[:, p[j+1] - 20:p[j+1]]) for j in range(len(p) - 1)]
            stop = 1
        all_goal_values = np.array(all_goal_values)


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
                     bbox_to_anchor=(0.5, 1.12),
                     ncol=7,
                     fancybox=True,
                     shadow=True,
                     prop={'size': 62, 'weight': 'bold'},
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

    # ax.hlines(y=0.93, xmin=0, xmax=(LAST_EP + 1) * NB_EPS_PER_EPOCH / 1000, color='black', linewidth=3, linestyles='dashed')
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    plt.grid()
    # ax.set_facecolor((244/255, 244/255, 244/255))
    save_fig(path=SAVE_PATH + folder + PLOT + '.pdf', artists=artists)
    return sr_per_cond_stats.copy()


if __name__ == '__main__':

    print('\n\tPlotting LP')
    experiment_path = RESULTS_PATH

    max_len, max_seeds, min_len, min_seeds = check_length_and_seeds(experiment_path=experiment_path)

    conditions = [f'main_beta={b}' for b in [50]]
    labels = [f'LP estimation']
    get_mean_sr(experiment_path, max_len, max_seeds, conditions, labels, ref=conditions[0])