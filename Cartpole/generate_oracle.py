import os
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import tyro
from causalpfn import CATEEstimator
import matplotlib.pyplot as plt

@dataclass
class Args:
    data_folder: str = "data/"
    env_id: str = "CartPole-v1"
    reward_mapping: Literal["original", "angle"] = "original"
    cuda: bool = True

if __name__ == "__main__":
    args = tyro.cli(Args)

    device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")

    # Load data
    with np.load(f"{args.data_folder}/{args.env_id}/random.npz") as data:
        X_random = data["states"].astype(np.float32)
        T_random = data["actions"].astype(np.float32)
        Y_random = np.concatenate([data["rewards"].astype(np.float32)[:, np.newaxis], data["next_states"].astype(np.float32), data["dones"].astype(np.float32)[:, np.newaxis]], axis=1)
        dones_random = data["dones"]
    with np.load(f"{args.data_folder}/{args.env_id}/dqn.npz") as data:
        X_dqn = data["states"].astype(np.float32)
        T_dqn = data["actions"].astype(np.float32)
        Y_dqn = np.concatenate([data["rewards"].astype(np.float32)[:, np.newaxis], data["next_states"].astype(np.float32), data["dones"].astype(np.float32)[:, np.newaxis]], axis=1)
        dones_dqn = data["dones"]
    X = np.concatenate([X_random, X_dqn], axis=0)
    T = np.concatenate([T_random, T_dqn], axis=0)
    Y = np.concatenate([Y_random, Y_dqn], axis=0)
    done = np.concatenate([dones_random, dones_dqn], axis=0)
    
    if args.reward_mapping == "angle":
        Y[:, 0] = np.cos(X[:, 2]) # pole angle

    print(f"Data loaded: n_samples={len(X)}, n_features={X.shape[1]}")

    target_names = ["Reward", "Cart_Position", "Cart_Velocity", "Pole_Angle", "Pole_Angular_Velocity", "Done"]
    states_random, next_states_random, actions_random, rewards_random, dones_random = X_random, np.ndarray((len(X_random), X_random.shape[1])), 1 - T_random, np.ndarray((len(X_random),)), np.ndarray((len(X_random),))
    states_dqn, next_states_dqn, actions_dqn, rewards_dqn, dones_dqn = X_dqn, np.ndarray((len(X_dqn), X_dqn.shape[1])), 1 - T_dqn, np.ndarray((len(X_dqn),)), np.ndarray((len(X_dqn),))

    context_idx = np.random.choice(np.count_nonzero(done), size=10000, replace=False)
    for i, name in enumerate(target_names):
        print(f"\nGenerating Oracle for target: {name}")
        causalpfn_cate = CATEEstimator(
            device=device,
            verbose=True,
        )
        
        if 0 < i < 5:
            valid_idx = np.where(done == 0)[0]
            X_i = X[valid_idx][context_idx]
            T_i = T[valid_idx][context_idx]
            Y_i = Y[valid_idx, i][context_idx]
        else:
            valid_idx = np.arange(len(X))
            X_i = X[context_idx]
            T_i = T[context_idx]
            Y_i = Y[context_idx, i]

        print(f"Valid samples for target {name}: {len(valid_idx)}")
        causalpfn_cate.fit(X_i, T_i, Y_i)
        print("CausalPFN CATE model trained.")

        cate_hat_random, mu_0_random, mu_1_random = causalpfn_cate.estimate_cate(X_random)
        cate_hat_dqn, mu_0_dqn, mu_1_dqn = causalpfn_cate.estimate_cate(X_dqn)
        Y_test_hat_random = mu_0_random * T_random + mu_1_random * (1 - T_random) # reverse treatment here
        Y_test_hat_dqn = mu_0_dqn * T_dqn + mu_1_dqn * (1 - T_dqn) # reverse treatment here
        if i == 0:
            rewards_random = Y_test_hat_random
            rewards_dqn = Y_test_hat_dqn
        elif i < 5:
            next_states_random[:, i - 1] = Y_test_hat_random
            next_states_dqn[:, i - 1] = Y_test_hat_dqn
        else:
            dones_random, dones_dqn = np.zeros_like(Y_test_hat_random), np.zeros_like(Y_test_hat_dqn)
            dones_random[Y_test_hat_random >= 0.5] = 1
            dones_dqn[Y_test_hat_dqn >= 0.5] = 1

    np.savez_compressed(
        f"data/{args.env_id}/random_oracle{'_remapped' if args.reward_mapping == 'angle' else ''}.npz",
        states_random=states_random,
        next_states_random=next_states_random,
        actions_random=actions_random,
        rewards_random=rewards_random,
        dones_random=dones_random,
    )

    np.savez_compressed(
        f"data/{args.env_id}/dqn_oracle{'_remapped' if args.reward_mapping == 'angle' else ''}.npz",
        states_dqn=states_dqn,
        next_states_dqn=next_states_dqn,
        actions_dqn=actions_dqn,
        rewards_dqn=rewards_dqn,
        dones_dqn=dones_dqn,
    )
