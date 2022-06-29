import torch
import numpy as np
import torch.nn.functional as F
import time
from mpi_utils.mpi_utils import sync_grads


def update_entropy(alpha, log_alpha, target_entropy, log_pi, alpha_optim, args):
    if args.automatic_entropy_tuning:
        # alpha_loss = -(log_alpha * (log_pi + target_entropy).detach()).mean()

        # alpha_optim.zero_grad()
        # alpha_loss.backward()
        # alpha_optim.step()

        # alpha = log_alpha.exp()
        # alpha_tlogs = alpha.clone()

        # Temporary for debug
        alpha_loss = torch.tensor(0.)
        alpha_tlogs = torch.tensor(alpha)
    else:
        alpha_loss = torch.tensor(0.)
        alpha_tlogs = torch.tensor(alpha)

    return alpha_loss, alpha_tlogs


def update_flat(actor_network, critic_network, critic_target_network, policy_optim, critic_optim, alpha, log_alpha, target_entropy,
                alpha_optim, obs_norm, ag_norm, g_norm, obs_next_norm, actions, rewards, args):
    inputs_norm = np.concatenate([obs_norm, ag_norm, g_norm], axis=1)
    inputs_next_norm = np.concatenate([obs_next_norm, ag_norm, g_norm], axis=1)

    # Tensorize
    inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
    inputs_next_norm_tensor = torch.tensor(inputs_next_norm, dtype=torch.float32)
    actions_tensor = torch.tensor(actions, dtype=torch.float32)
    r_tensor = torch.tensor(rewards, dtype=torch.float32).reshape(rewards.shape[0], 1)

    if args.cuda:
        inputs_norm_tensor = inputs_norm_tensor.cuda()
        inputs_next_norm_tensor = inputs_next_norm_tensor.cuda()
        actions_tensor = actions_tensor.cuda()
        r_tensor = r_tensor.cuda()

    with torch.no_grad():
        actions_next, log_pi_next, _ = actor_network.sample(inputs_next_norm_tensor)
        qf1_next_target, qf2_next_target = critic_target_network(inputs_next_norm_tensor, actions_next)
        min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * log_pi_next
        next_q_value = r_tensor + args.gamma * min_qf_next_target

    # the q loss
    qf1, qf2 = critic_network(inputs_norm_tensor, actions_tensor)
    qf1_loss = F.mse_loss(qf1, next_q_value)
    qf2_loss = F.mse_loss(qf2, next_q_value)

    # the actor loss
    pi, log_pi, _ = actor_network.sample(inputs_norm_tensor)
    qf1_pi, qf2_pi = critic_network(inputs_norm_tensor, pi)
    min_qf_pi = torch.min(qf1_pi, qf2_pi)
    policy_loss = ((alpha * log_pi) - min_qf_pi).mean()

    # start to update the network
    policy_optim.zero_grad()
    policy_loss.backward()
    sync_grads(actor_network)
    policy_optim.step()

    # update the critic_network
    critic_optim.zero_grad()
    qf1_loss.backward()
    sync_grads(critic_network)
    critic_optim.step()

    critic_optim.zero_grad()
    qf2_loss.backward()
    sync_grads(critic_network)
    critic_optim.step()

    alpha_loss, alpha_tlogs = update_entropy(alpha, log_alpha, target_entropy, log_pi, alpha_optim, args)

    return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()


def update_gnns(model, policy_optim, critic_optim, value_optim, alpha, log_alpha, target_entropy, alpha_optim, obs_norm, ag_norm, g_norm,
                    anchor_g_norm, obs_next_norm, ag_next_norm, actions, rewards, anchor_rewards, final_rewards, args):
    # Tensorize
    obs_norm_tensor = torch.tensor(obs_norm, dtype=torch.float32)
    obs_next_norm_tensor = torch.tensor(obs_next_norm, dtype=torch.float32)
    g_norm_tensor = torch.tensor(g_norm, dtype=torch.float32)
    ag_norm_tensor = torch.tensor(ag_norm, dtype=torch.float32)
    ag_next_norm_tensor = torch.tensor(ag_next_norm, dtype=torch.float32)
    actions_tensor = torch.tensor(actions, dtype=torch.float32)
    r_tensor = torch.tensor(rewards, dtype=torch.float32).reshape(rewards.shape[0], 1)

    anchor_g_norm_tensor = torch.tensor(anchor_g_norm, dtype=torch.float32)
    anchor_r_tensor = torch.tensor(anchor_rewards, dtype=torch.float32).reshape(anchor_rewards.shape[0], 1)
    final_r_tensor = torch.tensor(final_rewards, dtype=torch.float32).reshape(final_rewards.shape[0], 1)

    if args.cuda:
        obs_norm_tensor = obs_norm_tensor.cuda()
        obs_next_norm_tensor = obs_next_norm_tensor.cuda()
        g_norm_tensor = g_norm_tensor.cuda()
        ag_norm_tensor = ag_norm_tensor.cuda()
        ag_next_norm_tensor = ag_next_norm_tensor.cuda()
        actions_tensor = actions_tensor.cuda()
        r_tensor = r_tensor.cuda()

        anchor_g_norm_tensor = anchor_g_norm_tensor.cuda()
        anchor_r_tensor = anchor_r_tensor.cuda()
        final_r_tensor = final_r_tensor.cuda()

    with torch.no_grad():
        model.forward_pass(obs_next_norm_tensor, ag_next_norm_tensor, g_norm_tensor)
        actions_next, log_pi_next = model.pi_tensor, model.log_prob
        qf1_next_target, qf2_next_target = model.target_q1_pi_tensor, model.target_q2_pi_tensor
        min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * log_pi_next
        next_q_value = r_tensor + args.gamma * min_qf_next_target

    # the q loss
    qf1, qf2 = model.forward_pass(obs_norm_tensor, ag_norm_tensor, g_norm_tensor, actions=actions_tensor)
    qf1_loss = F.mse_loss(qf1, next_q_value)
    qf2_loss = F.mse_loss(qf2, next_q_value)
    qf_loss = qf1_loss + qf2_loss

    # the actor loss
    pi, log_pi = model.pi_tensor, model.log_prob
    qf1_pi, qf2_pi = model.q1_pi_tensor, model.q2_pi_tensor
    min_qf_pi = torch.min(qf1_pi, qf2_pi)
    policy_loss = ((alpha * log_pi) - min_qf_pi).mean()

    # Value loss
    # with torch.no_grad():
    #     model.forward_pass(obs_next_norm_tensor, ag_next_norm_tensor, anchor_g_norm_tensor)
    #     _, log_pi_n = model.pi_tensor, model.log_prob
    #     qf1_n_target, qf2_n_target = model.target_q1_pi_tensor, model.target_q2_pi_tensor
    #     min_qf_n_target = torch.min(qf1_n_target, qf2_n_target) - alpha * log_pi_n
    #     n_q_value = anchor_r_tensor + args.gamma * min_qf_n_target
    # v = model.value_forward_pass(ag_norm_tensor, anchor_g_norm_tensor)
    # v_loss = F.mse_loss(v, n_q_value)
    v = model.value_forward_pass(ag_norm_tensor, anchor_g_norm_tensor)
    v_loss = F.mse_loss(v, final_r_tensor)
    # start to update the network
    policy_optim.zero_grad()
    policy_loss.backward(retain_graph=True)
    sync_grads(model.actor)
    policy_optim.step()

    # update the critic_network
    critic_optim.zero_grad()
    qf_loss.backward()
    sync_grads(model.critic)
    critic_optim.step()

    # update the value network
    value_optim.zero_grad()
    v_loss.backward()
    sync_grads(model.value_network)
    value_optim.step()

    alpha_loss, alpha_tlogs = update_entropy(alpha, log_alpha, target_entropy, log_pi, alpha_optim, args)

    return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()
