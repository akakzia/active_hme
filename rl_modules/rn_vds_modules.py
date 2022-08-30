import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from itertools import permutations
import numpy as np
from rl_modules.networks import GnnMessagePassing, RhoActorDeepSet, RhoCriticDeepSet, RhoValueDeepSet, SelfAttention, VdRhoCriticDeepSet
from utils import get_graph_structure

epsilon = 1e-6

class ValueDisagreementGnn(nn.Module):
    def __init__(self, dim_rho_critic_input, dim_rho_critic_output):
        super(ValueDisagreementGnn, self).__init__()

        self.rho_vds = VdRhoCriticDeepSet(dim_rho_critic_input, dim_rho_critic_output)

    def forward(self, inp):
        return self.rho_vds(inp)

class RnCritic(nn.Module):
    def __init__(self, nb_objects, edges, incoming_edges, predicate_ids, dim_body, dim_object, dim_mp_input,
                 dim_mp_output, dim_rho_critic_input, dim_rho_critic_output):
        super(RnCritic, self).__init__()

        self.nb_objects = nb_objects
        self.dim_body = dim_body
        self.dim_object = dim_object

        self.n_permutations = self.nb_objects * (self.nb_objects - 1)

        self.mp_critic_1 = GnnMessagePassing(dim_mp_input, dim_mp_output)
        self.mp_critic_2 = GnnMessagePassing(dim_mp_output, dim_mp_output)
        self.edge_self_attention = SelfAttention(dim_mp_output, 1)
        self.rho_critic = RhoCriticDeepSet(dim_rho_critic_input, dim_rho_critic_output)

        self.edges = edges
        self.incoming_edges = incoming_edges
        self.predicate_ids = predicate_ids

    def forward(self, obs, act, ag, g):
        batch_size = obs.shape[0]
        assert batch_size == len(obs)

        # Critic message passing using node features, edge features and global features (here body + action)
        # Returns the edges features (which number is equal to the number of edges, i.e. permutations of objects)
        edge_features = self.message_passing(obs, act, ag, g)

        # Perform pooling (self-attention) on the edge features
        edge_features = edge_features.permute(1, 0, 2)
        edge_features_attention = self.edge_self_attention(edge_features)
        edge_features_attention = edge_features_attention.sum(dim=-2)

        # Readout function
        q1_pi_tensor, q2_pi_tensor = self.rho_critic(edge_features_attention, edge_features_attention)
        return q1_pi_tensor, q2_pi_tensor, edge_features_attention.detach()

    def message_passing(self, obs, act, ag, g):
        batch_size = obs.shape[0]
        assert batch_size == len(ag)
        
        obs_body = obs[:, :self.dim_body]
        obs_objects = [obs[:, self.dim_body + self.dim_object * i: self.dim_body + self.dim_object * (i + 1)]
                       for i in range(self.nb_objects)]

        inp_mp = torch.stack([torch.cat([obs_body, act, ag[:, self.predicate_ids[i]], g[:, self.predicate_ids[i]], obs_objects[self.edges[i][0]],
                                         obs_objects[self.edges[i][1]]], dim=-1) for i in range(self.n_permutations)])

        output_mp_1 = self.mp_critic_1(inp_mp)

        output_mp = self.mp_critic_2(output_mp_1)

        return output_mp


class RnActor(nn.Module):
    def __init__(self, nb_objects, edges, incoming_edges, predicate_ids, dim_body, dim_object, dim_mp_input, dim_mp_output, 
                 dim_rho_actor_input,dim_rho_actor_output):
        super(RnActor, self).__init__()

        self.nb_objects = nb_objects
        self.dim_body = dim_body
        self.dim_object = dim_object

        self.n_permutations = self.nb_objects * (self.nb_objects - 1)

        self.mp_actor_1 = GnnMessagePassing(dim_mp_input, dim_mp_output)
        self.mp_actor_2 = GnnMessagePassing(dim_mp_output, dim_mp_output)
        self.edge_self_attention = SelfAttention(dim_mp_output, 1)
        self.rho_actor = RhoActorDeepSet(dim_rho_actor_input, dim_rho_actor_output)

        self.edges = edges
        self.incoming_edges = incoming_edges
        self.predicate_ids = predicate_ids
    
    def message_passing(self, obs, ag, g):
        batch_size = obs.shape[0]
        assert batch_size == len(ag)

        obs_body = obs[:, :self.dim_body]
        obs_objects = [obs[:, self.dim_body + self.dim_object * i: self.dim_body + self.dim_object * (i + 1)]
                       for i in range(self.nb_objects)]

        # delta_g = g - ag

        inp_mp = torch.stack([torch.cat([obs_body, ag[:, self.predicate_ids[i]], g[:, self.predicate_ids[i]], obs_objects[self.edges[i][0]],
                                         obs_objects[self.edges[i][1]]], dim=-1) for i in range(self.n_permutations)])

        output_mp_1 = self.mp_actor_1(inp_mp)

        output_mp = self.mp_actor_2(output_mp_1)

        return output_mp

    def forward(self, obs, ag, g):
        batch_size = obs.shape[0]
        assert batch_size == len(obs)

        # Actor message passing using node features, edge features and global features (here body)
        # Returns the edges features (which number is equal to the number of edges, i.e. permutations of objects)
        edge_features = self.message_passing(obs, ag, g)

        # Perform pooling (self-attention) on the edge features
        edge_features = edge_features.permute(1, 0, 2)
        edge_features_attention = self.edge_self_attention(edge_features)
        edge_features_attention = edge_features_attention.sum(dim=-2)

        # Readout function
        mean, logstd = self.rho_actor(edge_features_attention)

        return mean, logstd

    def sample(self, obs, ag, g):
        mean, log_std = self.forward(obs, ag, g)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log((1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob, torch.tanh(mean)


class RnGoalValue(nn.Module):
    def __init__(self, nb_objects, edges, incoming_edges, predicate_ids, cuda):
        super(RnGoalValue, self).__init__()

        self.nb_objects = nb_objects

        dim_mp_input = 2 * (self.nb_objects + 4) # 2 * (object dim (one hot) + nb_predicates per pair)
        dim_mp_output = 3 * dim_mp_input

        dim_rho_value_input = dim_mp_output
        dim_rho_value_output = 1

        self.n_permutations = self.nb_objects * (self.nb_objects - 1)

        self.mp_value_1 = GnnMessagePassing(dim_mp_input, dim_mp_output)
        self.mp_value_2 = GnnMessagePassing(dim_mp_output, dim_mp_output)
        self.edge_self_attention = SelfAttention(dim_mp_output, 1)
        self.rho_value = RhoValueDeepSet(dim_rho_value_input, dim_rho_value_output)

        self.edges = edges
        self.incoming_edges = incoming_edges
        self.predicate_ids = predicate_ids

        self.args_cuda = cuda

    def forward(self, ag, g):
        # Value message passing using node features, edge features and global features (here body + action)
        # Returns the edges features (which number is equal to the number of edges, i.e. permutations of objects)
        edge_features = self.message_passing(ag, g)


        # Perform pooling (self-attention) on the edge features
        edge_features = edge_features.permute(1, 0, 2)
        edge_features_attention = self.edge_self_attention(edge_features)
        edge_features_attention = edge_features_attention.sum(dim=-2)

        # Readout function
        v_tensor= self.rho_value(edge_features_attention)
        return v_tensor

    def message_passing(self, ag, g):
        batch_size = g.shape[0]
        objects_one_hot = [torch.Tensor([1., 0., 0., 0., 0.]), torch.Tensor([0., 1., 0., 0., 0.]), 
                           torch.Tensor([0., 0., 1., 0., 0.]), torch.Tensor([0., 0., 0., 1., 0.]), 
                           torch.Tensor([0., 0., 0., 0., 1.])]
        
        if self.args_cuda:
            objects_one_hot = [e.unsqueeze(0).repeat(batch_size, 1).cuda() for e in objects_one_hot]
        else:
            objects_one_hot = [e.unsqueeze(0).repeat(batch_size, 1) for e in objects_one_hot]

        inp_mp = torch.stack([torch.cat([ag[:, self.predicate_ids[i]], g[:, self.predicate_ids[i]], objects_one_hot[self.edges[i][0]],
                                         objects_one_hot[self.edges[i][1]]], dim=-1) for i in range(self.n_permutations)])

        output_mp_1 = self.mp_value_1(inp_mp)

        output_mp = self.mp_value_2(output_mp_1)

        return output_mp

class RnSemantic:
    def __init__(self, env_params, args):
        self.dim_body = 10
        self.dim_object = 15
        self.dim_goal = env_params['goal']
        self.dim_act = env_params['action']
        self.nb_objects = args.n_blocks

        self.q1_pi_tensor = None
        self.q2_pi_tensor = None
        self.target_q1_pi_tensor = None
        self.target_q2_pi_tensor = None
        self.pi_tensor = None
        self.log_prob = None
        self.value = None

        # Process indexes for graph construction
        self.edges, self.incoming_edges, self.predicate_ids = get_graph_structure(self.nb_objects)

        dim_mp_actor_input = 2 * (self.dim_object + 4) + self.dim_body # 2 * dim node + dim partial goal + dim global
        dim_mp_actor_output = 3 * dim_mp_actor_input

        dim_mp_critic_input = 2 * (self.dim_object + 4) + (self.dim_body + self.dim_act) # 2 * dim node + dim partial goal + dim global
        dim_mp_critic_output = 3 * dim_mp_actor_input

        dim_rho_actor_input = dim_mp_actor_output
        dim_rho_actor_output = self.dim_act

        dim_rho_critic_input = dim_mp_critic_output
        dim_rho_critic_output = 1

        self.critic = RnCritic(self.nb_objects, self.edges, self.incoming_edges, self.predicate_ids,
                                self.dim_body, self.dim_object, dim_mp_critic_input, dim_mp_critic_output,
                                dim_rho_critic_input, dim_rho_critic_output)
        self.critic_target = RnCritic(self.nb_objects, self.edges, self.incoming_edges, self.predicate_ids,
                                       self.dim_body, self.dim_object, dim_mp_critic_input, dim_mp_critic_output,
                                       dim_rho_critic_input, dim_rho_critic_output)
        
        if args.agent == 'VDSAgent':
            self.vds_q_values = ValueDisagreementGnn(dim_rho_critic_input, dim_rho_critic_output)
            self.vds_target_q_values = ValueDisagreementGnn(dim_rho_critic_input, dim_rho_critic_output)

        self.actor = RnActor(self.nb_objects, self.edges, self.incoming_edges, self.predicate_ids, self.dim_body, self.dim_object, 
                              dim_mp_actor_input, dim_mp_actor_output, dim_rho_actor_input, dim_rho_actor_output)
        
        self.value_network = RnGoalValue(self.nb_objects, self.edges, self.incoming_edges, self.predicate_ids, args.cuda)

    def policy_forward_pass(self, obs, ag, g, no_noise=False):
        if not no_noise:
            self.pi_tensor, self.log_prob, _ = self.actor.sample(obs, ag, g)
        else:
            _, self.log_prob, self.pi_tensor = self.actor.sample(obs, ag, g)

    def forward_pass(self, obs, ag, g, actions=None):
        self.pi_tensor, self.log_prob, _ = self.actor.sample(obs, ag, g)

        if actions is not None:
            self.q1_pi_tensor, self.q2_pi_tensor, input_vds = self.critic.forward(obs, self.pi_tensor, ag, g)
            self.q1_vd_tensor, self.q2_vd_tensor, self.q3_vd_tensor = self.vds_q_values.forward(input_vds)
            return self.critic.forward(obs, actions, ag, g)
        else:
            with torch.no_grad():
                self.target_q1_pi_tensor, self.target_q2_pi_tensor, input_vds = self.critic_target.forward(obs, self.pi_tensor, ag, g)
                self.target_q1_vd_tensor, self.target_q2_vd_tensor, self.target_q3_vd_tensor = self.vds_target_q_values.forward(input_vds)
            self.q1_pi_tensor, self.q2_pi_tensor = None, None
    
    def value_forward_pass(self, ag, g):
        self.value = self.value_network.forward(ag, g)

        return self.value