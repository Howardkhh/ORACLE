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
    data_type: Literal["random", "dqn", "mix"]
    data_folder: str = "data/"
    env_id: str = "CartPole-v1"
    reward_mapping: Literal["original", "angle"] = "original"
    cuda: bool = True
    validation_split: float = 0.3

if __name__ == "__main__":
    args = tyro.cli(Args)

    device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")

    # Load data
    if args.data_type == "random" or args.data_type == "dqn":
        with np.load(f"{args.data_folder}/{args.env_id}/{args.data_type}.npz") as data:
            X = data["states"].astype(np.float32)
            T = data["actions"].astype(np.float32)
            Y = np.concatenate([data["rewards"].astype(np.float32)[:, np.newaxis], data["next_states"].astype(np.float32), data["dones"].astype(np.float32)[:, np.newaxis]], axis=1)
            done = data["dones"]
    elif args.data_type == "mix":
        with np.load(f"{args.data_folder}/{args.env_id}/random.npz") as data:
            X_random = data["states"].astype(np.float32)
            T_random = data["actions"].astype(np.float32)
            Y_random = np.concatenate([data["rewards"].astype(np.float32)[:, np.newaxis], data["next_states"].astype(np.float32), data["dones"].astype(np.float32)[:, np.newaxis]], axis=1)
            done_random = data["dones"]
        with np.load(f"{args.data_folder}/{args.env_id}/dqn.npz") as data:
            X_dqn = data["states"].astype(np.float32)
            T_dqn = data["actions"].astype(np.float32)
            Y_dqn = np.concatenate([data["rewards"].astype(np.float32)[:, np.newaxis], data["next_states"].astype(np.float32), data["dones"].astype(np.float32)[:, np.newaxis]], axis=1)
            done_dqn = data["dones"]
        X = np.concatenate([X_random, X_dqn], axis=0)
        T = np.concatenate([T_random, T_dqn], axis=0)
        Y = np.concatenate([Y_random, Y_dqn], axis=0)
        done = np.concatenate([done_random, done_dqn], axis=0)
    else:
        raise ValueError("Unknown data type")
    
    if args.reward_mapping == "angle":
        Y[:, 0] = np.cos(X[:, 2]) # pole angle

    print(f"Data loaded: n_samples={len(X)}, n_features={X.shape[1]}")

    n = len(X) // 10 # 100000
    if args.data_type == "mix": n //= 2
    test_size = int(n * args.validation_split) # 30000
    data_sizes = [n, n // 2, n // 10, n // 20, n // 100]

    target_names = ["Reward", "Cart_Position", "Cart_Velocity", "Pole_Angle", "Pole_Angular_Velocity", "Done"]
    for i, name in enumerate(target_names):
        print(f"\nEstimating CATE for target: {name}")

        plt.figure(figsize=(10, 10))
        plt.hist2d(T, Y[:, i], bins=(2, 100), cmin=1)
        plt.xlabel("Treatment")
        plt.ylabel(name)
        plt.title(f"{name} Distribution ({args.data_type} data" + f", reward mapping: {args.reward_mapping})" if i == 0 else ")")
        plt.colorbar(label="Count")
        os.makedirs(f"outputs/{args.env_id}", exist_ok=True)
        plt.savefig(f"outputs/{args.env_id}/{name}_distribution_{args.data_type}_{args.reward_mapping}.png")

        plt.figure(figsize=(12, 8))
        if 0 < i < 5:
            valid_idx = np.where(done == 0)[0]
        else:
            valid_idx = np.arange(len(X))
        print(f"Valid samples for target {name}: {len(valid_idx)}")
        test_idx = np.random.choice(valid_idx, size=test_size, replace=False)
        for idx, data_size in enumerate(data_sizes):
            train_idx = np.random.choice(np.setdiff1d(valid_idx, test_idx), size=data_size, replace=False)
            X_train, X_test = X[train_idx], X[test_idx]
            T_train, T_test = T[train_idx], T[test_idx]
            Y_train, Y_test = Y[train_idx, i], Y[test_idx, i]

            print(f"Training CausalPFN with data size: {data_size}")
            causalpfn_cate = CATEEstimator(
                device=device,
                verbose=True,
            )
            causalpfn_cate.fit(X_train, T_train, Y_train)
            print("CausalPFN CATE model trained.")
            cate_hat, mu_0, mu_1 = causalpfn_cate.estimate_cate(X_test)
            Y_test_hat = mu_0 * (1 - T_test) + mu_1 * T_test

            if i < 5:
                rmse = np.sqrt(np.mean((Y_test_hat - Y_test) ** 2))
                print(f"Data size: {data_size}, Test RMSE: {rmse}")
            else:
                Y_test_pred = np.zeros_like(Y_test_hat)
                Y_test_pred[Y_test_hat > 0.5] = 1
                accuracy = np.mean((Y_test_pred == Y_test))
                print(f"Data size: {data_size}, Test Accuracy: {accuracy}")

            plt.subplot(2, 3, idx + 1)
            plt.scatter(mu_0[T_test == 0], Y_test[T_test == 0], alpha=0.1, label="T=0", color='blue', s=1)
            plt.scatter(mu_1[T_test == 1], Y_test[T_test == 1], alpha=0.1, label="T=1", color='orange', s=1)
            mini, maxi = np.min(Y_test), np.max(Y_test)
            if i < 5: plt.plot([mini, maxi], [mini, maxi], 'k--', alpha=0.5)
            plt.xlabel(f"Predicted {name}")
            plt.ylabel(f"True {name}")
            plt.title(f"Data size: {data_size}, {'RMSE' if i < 5 else 'Accuracy'}: {rmse if i < 5 else accuracy:.8f}")
            plt.legend()
        
        if i == 0:
            Y_test_hat = 1 - X_test[:, 2] ** 2 / 2  # approximate cos(x) = 1 - x^2/2 for baseline
        elif i == 5:
            Y_test_hat = np.zeros_like(Y_test)  # always not done for baseline
        else:
            Y_test_hat = X_test[:, i - 1]  # same next state for baseline

        if i < 5:
            rmse = np.sqrt(np.mean((Y_test_hat - Y_test) ** 2))
            print(f"Baseline Test RMSE: {rmse}")
        else:
            accuracy = np.mean((Y_test_hat == Y_test))
            print(f"Baseline Test Accuracy: {accuracy}")
        plt.subplot(2, 3, len(data_sizes) + 1)
        plt.scatter(Y_test_hat, Y_test, alpha=0.1, label="Baseline", color='green', s=1)
        mini, maxi = np.min(Y_test), np.max(Y_test)
        plt.plot([mini, maxi], [mini, maxi], 'k--', alpha=0.5)
        plt.xlabel(f"Predicted {name}")
        plt.ylabel(f"True {name}")
        plt.title(f"Data size: {data_size}, {'RMSE' if i < 5 else 'Accuracy'}: {rmse if i < 5 else accuracy:.8f}")
        plt.legend()


        plt.tight_layout()
        plt.savefig(f"outputs/{args.env_id}/causalpfn_{name}_predictions_{args.data_type}_{args.reward_mapping}.png")