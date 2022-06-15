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
    parser.add_argument('--num-workers', type=int, default=MPI.COMM_WORLD.Get_size(), help='the number of cpus to collect samples')
    parser.add_argument('--cuda', action='store_true', help='if use gpu do the acceleration')
    # the environment arguments
    parser.add_argument('--algo', type=str, default='semantic', help="'semantic'")
    parser.add_argument('--agent', type=str, default='SAC', help='the RL algorithm name')
    parser.add_argument('--n-blocks', type=int, default=5, help='The number of blocks to be considered in the FetchManipulate env')
    # the training arguments
    parser.add_argument('--n-epochs', type=int, default=1000, help='the number of epochs to train the agent')
    parser.add_argument('--n-cycles', type=int, default=50, help='the times to collect samples per epoch')
    parser.add_argument('--n-batches', type=int, default=30, help='the times to update the network')
    parser.add_argument('--num-rollouts-per-mpi', type=int, default=2, help='the rollouts per mpi')
    parser.add_argument('--batch-size', type=int, default=256, help='the sample batch size')
    # the replay arguments
    parser.add_argument('--multi-criteria-her', type=bool, default=True, help='test')
    parser.add_argument('--replay-strategy', type=str, default='future', help='the HER strategy')
    parser.add_argument('--replay-k', type=int, default=1, help='ratio to be replace')
    parser.add_argument('--reward-type', type=str, default='per_object', help='per_object, per_relation, per_predicate or sparse')
    # The RL arguments
    parser.add_argument('--self-eval-prob', type=float, default=0.1, help='Probability to perform self-evaluation')
    parser.add_argument('--gamma', type=float, default=0.98, help='the discount factor')
    parser.add_argument('--alpha', type=float, default=0.2, help='entropy coefficient')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, help='Tune entropy')
    parser.add_argument('--action-l2', type=float, default=1, help='l2 reg')
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
    parser.add_argument('--normalize_goal', type=bool, default=False, help='do evaluation at the end of the epoch w/ frequency')
    parser.add_argument('--clip-range', type=float, default=5, help='the clip range')
    # the gnns arguments
    parser.add_argument('--architecture', type=str, default='relation_network', help='[full_gn, interaction_network, relation_network, deep_sets, flat]')
    # the testing arguments
    parser.add_argument('--n-test-rollouts', type=int, default=1, help='the number of tests')

    # the goal evaluator arguments
    parser.add_argument('--goal-evaluator-method', type=int, default=1, help='either 1 (neural net estimation) or 2 (policy evaluation)')
    parser.add_argument('--start-curriculum-k', type=int, default=1000000, help='Start generate buckets after k cycles')
    parser.add_argument('--bucket-generation-freq', type=int, default=250, help='Generate buckets every n cycles')
    parser.add_argument('--bucket-evaluation-freq', type=int, default=50, help='Evaluate values of goals every m cycles')
    parser.add_argument('--granularity', type=int, default=3, help='Maximum number of buckets that can be created')
    parser.add_argument('--normalization-technique', type=str, default='mixed', help='[linear_fixed, linear_moving, mixed]')
    parser.add_argument('--epsilon-curriculum', type=float, default=0.1, help='Probability to perform random exploration of discovered goals')
    parser.add_argument('--use-stability-condition', type=bool, default=False, help='only consider stable goals as discovered')
    
    parser.add_argument('--strategy', type=int, default=2, help='Possible values: 0: Frontier; 1: Frontier and Stop, 2: Frontier and Beyond'
                                                                   '3: Beyond')
    
    parser.add_argument('--data-augmentation', type=bool, default=True, help='Augment guided episodes by relabeling')

    args = parser.parse_args()

    return args
