import numpy as np
import torch

from causalpfn import CATEEstimator


def simulate(args, states, actions):
    masscart, masspole, length, gravity, tau = 1.0, 0.1, 0.5, 9.8, 0.02
    x_threshold = 2.4
    theta_threshold_radians = 12 * 2 * np.pi / 360
    total_mass = masspole + masscart
    polemass_length = masspole * length # masspole * length
    x, x_dot, theta, theta_dot = states[:, 0], states[:, 1], states[:, 2], states[:, 3]
    force = np.where(actions == 1, 10.0, -10.0)
    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    temp = (
        force + polemass_length * theta_dot**2 * sintheta
    ) / total_mass
    thetaacc = (gravity * sintheta - costheta * temp) / (
        length * (4.0 / 3.0 - masspole * costheta**2 / total_mass)
    )
    xacc = temp - polemass_length * thetaacc * costheta / total_mass

    x = x + tau * x_dot
    x_dot = x_dot + tau * xacc
    theta = theta + tau * theta_dot
    theta_dot = theta_dot + tau * thetaacc

    next_states = np.stack([x, x_dot, theta, theta_dot], axis=1)
    dones = (x < -x_threshold) | (x > x_threshold) | (theta < -theta_threshold_radians) | (theta > theta_threshold_radians)
    if args.reward_mapping == "original":
        rewards = 1 # always 1 reward per step
    elif args.reward_mapping == "angle":
        rewards = np.cos(theta)
        rewards[dones] = np.cos(12/180*np.pi)  # terminal state angle is set to 12 degrees

    return next_states, rewards, dones

def estimate_with_causalpfn(args, X, T, Y):
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    num_outcomes = Y.shape[1]
    Y_hat = []
    for i in range(num_outcomes):
        causalpfn_cate = CATEEstimator(
            device=device,
            verbose=False,
        )
        causalpfn_cate.fit(X, 1-T, Y[:, i]) # reverse treatment here
        cate_hat, mu_0, mu_1 = causalpfn_cate.estimate_cate(X)
        Y_hat_i = mu_0 * (1 - T) + mu_1 * T
        Y_hat.append(Y_hat_i)
        
    rewards_hat = Y_hat[0]
    next_states_hat = np.stack(Y_hat[1:-1], axis=1)
    dones_hat = Y_hat[-1] >= 0.5

    return next_states_hat, rewards_hat, dones_hat
    

def get_oracle(args, states, actions, rewards, next_states, dones):
    actions = 1 - actions
    if args.oracle_source == "simulator":
        next_states, rewards, dones = simulate(args, states, actions)
    elif args.oracle_source == "causalpfn":
        next_states, rewards, dones = estimate_with_causalpfn(args, states, actions, np.concatenate([rewards[:, np.newaxis], next_states, dones[:, np.newaxis]], axis=1))
    return actions, next_states, rewards, dones

def load_data(args, oracle, idx=None):
    # load data from random policy
    with np.load(f"{args.data_folder}/{args.env_id}/random{'_remapped' if oracle and args.reward_mapping == 'angle' else ''}.npz") as data:
        states_random = data["states"].astype(np.float32)
        actions_random = data["actions"].astype(np.float32)
        rewards_random = data["rewards"].astype(np.float32)
        next_states_random = data["next_states"].astype(np.float32)
        dones_random = data["dones"].astype(bool)
    if oracle:
        actions_random, next_states_random, rewards_random, dones_random = get_oracle(args, states_random, actions_random, rewards_random, next_states_random, dones_random)

    # load data from dqn policy
    with np.load(f"{args.data_folder}/{args.env_id}/dqn{'_remapped' if oracle and args.reward_mapping == 'angle' else ''}.npz") as data:
        states_dqn = data["states"].astype(np.float32)
        actions_dqn = data["actions"].astype(np.float32)
        rewards_dqn = data["rewards"].astype(np.float32)
        next_states_dqn = data["next_states"].astype(np.float32)
        dones_dqn = data["dones"].astype(bool)
    if oracle:
        actions_dqn, next_states_dqn, rewards_dqn, dones_dqn = get_oracle(args, states_dqn, actions_dqn, rewards_dqn, next_states_dqn, dones_dqn)

    assert states_random.shape[0] == states_dqn.shape[0], "random and dqn data must have the same number of samples to mix (not really but it should be)"
    
    data_size = args.num_samples // 2 if args.oracle == "both" else args.num_samples

    if idx:
        idx_random, idx_dqn = idx
    else:
        num_random, num_dqn = round(data_size * args.method_ratio), round(data_size * (1 - args.method_ratio))
        rng = np.random.default_rng(data_size)
        idx_random, idx_dqn = rng.choice(states_random.shape[0], size=num_random, replace=False), rng.choice(states_dqn.shape[0], size=num_dqn, replace=False)
        idx = (idx_random, idx_dqn)
    states_random, actions_random, rewards_random, next_states_random, dones_random = states_random[idx_random], actions_random[idx_random], rewards_random[idx_random], next_states_random[idx_random], dones_random[idx_random]
    states_dqn, actions_dqn, rewards_dqn, next_states_dqn, dones_dqn = states_dqn[idx_dqn], actions_dqn[idx_dqn], rewards_dqn[idx_dqn], next_states_dqn[idx_dqn], dones_dqn[idx_dqn]
    states = np.concatenate([states_random, states_dqn], axis=0)
    actions = np.concatenate([actions_random, actions_dqn], axis=0)
    rewards = np.concatenate([rewards_random, rewards_dqn], axis=0)
    next_states = np.concatenate([next_states_random, next_states_dqn], axis=0)
    dones = np.concatenate([dones_random, dones_dqn], axis=0)
    return states, actions, rewards, next_states, dones, idx
