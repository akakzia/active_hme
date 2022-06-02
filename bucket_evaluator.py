import enum
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
from utils import generate_stacks_dict
from utils import CompressPDF

font = {'size': 60}
matplotlib.rc('font', **font)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams['figure.constrained_layout.use'] = True

cmap = plt.get_cmap('tab20b')
colors = np.array(cmap.colors)[[0, 17, 2, 16, 5, 8, 9, 12, 13, 14, 16, 17, 18]]
DPI = 30
COMPRESSOR = CompressPDF(4)

RESULTS_PATH = '/home/ahmed/Documents/Amaterasu/hachibi/active_hme/results/curriculum_study_buckets/'
SAVE_PATH = '/home/ahmed/Documents/Amaterasu/hachibi/active_hme/plots/'

def setup_n_figs(n, m):
    fig, axs = plt.subplots(n, m, figsize=(m * 18, n * 18), frameon=False)
    # axs = axs.ravel()
    artists = ()
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


if __name__ == '__main__':
    ep = 210
    gr = 3
    
    fig, artists, ax = setup_n_figs(n=1, m=gr)

    experiment_path = RESULTS_PATH + f'method=1_granularity={gr}_norm=mixed_startk=1000'
    list_runs = sorted(os.listdir(experiment_path))

    # Get evaluation map that correspond each goal to a class involving stacks, based only on the above predicates
    # This map ignores the close predicates
    # i.e. if there is only a stack of 2 blocks, we ignore the close predicates
    # This is used to check what types of goals the agent is discovering
    stacks_classes = ['stack_2', 'stack_3', '2stacks_2_2', '2stacks_2_3', 'pyramid_3', 'mixed_2_3', 'stack_4', 'stack_5']
    classes_to_ids = {k:i for i, k in enumerate(stacks_classes)}
    n_classes = len(stacks_classes)

    stacks_to_class = generate_stacks_dict(list_classes=stacks_classes, n_blocks=5, n_trials=2000)

    stacks_classes += ['others']
    classes_to_ids['others'] = n_classes
    
    for run in list_runs:
        model_path = experiment_path + '/' + run
        with open(model_path + f'/buckets/bucket_ep_{ep}.pkl', 'rb') as f:
            data_buckets = pkl.load(file=f)
        
        with open(model_path + f'/buckets/discovered_g_ep_{ep}.pkl', 'rb') as f:
            data_discovered = pkl.load(file=f)
        
        # for each bucket determine the repartition of goals according t the stacks class
        for k, v in data_buckets.items():
            goals_per_class = np.zeros(n_classes + 1)
            for goal_id in v:
                try:
                    c = stacks_to_class[str(data_discovered[goal_id][10:])]
                    goals_per_class[classes_to_ids[c]] += 1
                except KeyError:
                    goals_per_class[classes_to_ids['others']] += 1
            y = goals_per_class.astype(np.int)

            ax[k].pie(y, colors=colors)
            ax[k].set_title(f'Bucket {k}')
        leg = fig.legend(stacks_classes,
                    loc='upper center',
                    bbox_to_anchor=(0.525, 1.22),
                    ncol=9,
                    fancybox=True,
                    shadow=True,
                    prop={'size': 80, 'weight': 'normal'},
                    markerscale=1)
        artists += (leg,)

        save_fig(path=SAVE_PATH + f'granularity={gr}_buckets={ep}.pdf', artists=artists)