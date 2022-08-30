import argparse
import numpy as np
from mpi4py import MPI


"""
Here are the param for the training

"""


def get_args():
    parser = argparse.ArgumentParser()
    # the general arguments
    parser.add_argument('--seed', type=int, default=np.random.randint(1e6), help='random seed')
    parser.add_argument('--num-workers', type=int, default=MPI.COMM_WORLD.Get_size(), help='number of cpus to collect samples')
    parser.add_argument('--cuda', action='store_true', help='if use gpu do the acceleration')
    parser.add_argument('--agent', type=str, default='HME', help='goal exploration process to be used')
    # the environment arguments
    parser.add_argument('--n-blocks', type=int, default=5, help='number of blocks to be considered in the FetchManipulate env')
    # the training arguments
    parser.add_argument('--n-epochs', type=int, default=200, help='number of epochs to train the agent')
    parser.add_argument('--n-cycles', type=int, default=50, help='times to collect samples per epoch')
    parser.add_argument('--n-batches', type=int, default=30, help='times to update the network')
    parser.add_argument('--num-rollouts-per-mpi', type=int, default=2, help='rollouts per mpi')
    parser.add_argument('--batch-size', type=int, default=256, help='sample batch size')
    # the replay arguments
    parser.add_argument('--multi-criteria-her', type=bool, default=True, help='use multi-criteria HER')
    parser.add_argument('--replay-strategy', type=str, default='future', help='HER strategy to be used')
    parser.add_argument('--replay-k', type=int, default=1, help='ratio to replace goals')
    # The RL argumentss
    parser.add_argument('--gamma', type=float, default=0.98, help='the discount factor')
    parser.add_argument('--alpha', type=float, default=0.2, help='entropy coefficient')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, help='Tune entropy')
    parser.add_argument('--lr-actor', type=float, default=0.001, help='the learning rate of the actor')
    parser.add_argument('--lr-critic', type=float, default=0.001, help='the learning rate of the critic')
    parser.add_argument('--lr-entropy', type=float, default=0.001, help='the learning rate of the entropy')
    parser.add_argument('--polyak', type=float, default=0.95, help='the average coefficient')
    parser.add_argument('--freq-target_update', type=int, default=2, help='the frequency of updating the target networks')
    # the output arguments
    parser.add_argument('--evaluations', type=bool, default=True, help='do evaluation at the end of the epoch w/ frequency')
    parser.add_argument('--save-freq', type=int, default=10, help='the interval that save the trajectory')
    parser.add_argument('--save-dir', type=str, default='output/', help='the path to save the models')
    # the memory arguments
    parser.add_argument('--buffer-size', type=int, default=int(1e6), help='the size of the buffer')
    # the preprocessing arguments
    parser.add_argument('--clip-obs', type=float, default=5, help='the clip ratio')
    parser.add_argument('--clip-range', type=float, default=5, help='the clip range')
    # the gnns arguments
    parser.add_argument('--architecture', type=str, default='relation_network', help='[full_gn, interaction_network, relation_network, deep_sets, flat]')
    # the testing arguments
    parser.add_argument('--n-test-rollouts', type=int, default=1, help='the number of tests')

    # the goal evaluator arguments
    parser.add_argument('--normalization-technique', type=str, default='linear_fixed', help='[linear_fixed, linear_moving, mixed]')
    parser.add_argument('--use-stability-condition', type=bool, default=True, help='only consider stable goals as discovered')
    
    parser.add_argument('--data-augmentation', type=bool, default=True, help='Augment guided episodes by relabeling')

    parser.add_argument('--min-queue-length', type=int, default=50, help='minimum queue length to update query proba')
    parser.add_argument('--max-queue-length', type=int, default=200, help='maximum window of query proba update')
    parser.add_argument('--beta', type=int, default=50, help='sensitivity to social signals. 0: Social; 500: Autotelic')
    parser.add_argument('--progress-function', type=str, default='mean', help='aggregation function to compute query proba')

    parser.add_argument('--autotelic-planning-proba', type=float, default=0., help='probability to perform planning')

    parser.add_argument('--oracle-path', type=str, default='data/', help='path to SP model of the goal space')
    parser.add_argument('--oracle-name', type=str, default='oracle_perm_block', help='the nature of assignment space to be used')

    parser.add_argument('--apply-her-on-social', type=bool, default=True, help='when social episode, either to replay with HER')

    parser.add_argument('--internalization-strategy', type=int, default=1, help='0: None; 1, 2, 3, 4 to be described later')

    parser.add_argument('--n-freeplay-epochs', type=int, default=5, help='number of epochs where agents must perform autotelic episodes')

    parser.add_argument('--query-proba-update-freq', type=int, default=300, help='In how many episodes update the query proba')

    parser.add_argument('--fixed-queries', type=bool, default=False, help='either to perform fixed queries or active ones')
    parser.add_argument('--fixed-query-proba', type=float, default=0.1, help='fixed query proba, ignored if using active queries')

    args = parser.parse_args()

    return args
