import os
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import tyro
from causalpfn import CATEEstimator
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

@dataclass
class Args:
    data_type: Literal["random", "dqn", "mix"]
    data_folder: str = "data/"
    env_id: str = "CartPole-v1"
    reward_mapping: Literal["original", "angle"] = "angle"
    cuda: bool = True

def load_data(args: Args, validation: bool):
    if args.data_type == "random" or args.data_type == "dqn":
        with np.load(f"{args.data_folder}/{args.env_id}/{args.data_type}{'_remapped' if args.reward_mapping == 'angle' else ''}{'_validation' if validation else ''}.npz") as data:
            X = data["states"].astype(np.float32)
            T = data["actions"].astype(np.float32)
            Y = np.concatenate([data["rewards"].astype(np.float32)[:, np.newaxis], data["next_states"].astype(np.float32), data["dones"].astype(np.float32)[:, np.newaxis]], axis=1)
            done = data["dones"]
    elif args.data_type == "mix":
        with np.load(f"{args.data_folder}/{args.env_id}/random{'_remapped' if args.reward_mapping == 'angle' else ''}{'_validation' if validation else ''}.npz") as data:
            X_random = data["states"].astype(np.float32)
            T_random = data["actions"].astype(np.float32)
            Y_random = np.concatenate([data["rewards"].astype(np.float32)[:, np.newaxis], data["next_states"].astype(np.float32), data["dones"].astype(np.float32)[:, np.newaxis]], axis=1)
            done_random = data["dones"]
        with np.load(f"{args.data_folder}/{args.env_id}/dqn{'_remapped' if args.reward_mapping == 'angle' else ''}{'_validation' if validation else ''}.npz") as data:
            X_dqn = data["states"].astype(np.float32)
            T_dqn = data["actions"].astype(np.float32)
            Y_dqn = np.concatenate([data["rewards"].astype(np.float32)[:, np.newaxis], data["next_states"].astype(np.float32), data["dones"].astype(np.float32)[:, np.newaxis]], axis=1)
            done_dqn = data["dones"]

        rand_idx1 = np.random.randint(0, len(X_random), size=len(X_random) // 2)
        rand_idx2 = np.random.randint(0, len(X_dqn), size=len(X_random) // 2)
        
        X = np.concatenate([X_random[rand_idx1], X_dqn[rand_idx2]], axis=0)
        T = np.concatenate([T_random[rand_idx1], T_dqn[rand_idx2]], axis=0)
        Y = np.concatenate([Y_random[rand_idx1], Y_dqn[rand_idx2]], axis=0)
        done = np.concatenate([done_random[rand_idx1], done_dqn[rand_idx2]], axis=0)
    else:
        raise ValueError("Unknown data type")
    
    return X, T, Y, done
    

if __name__ == "__main__":
    args = tyro.cli(Args)

    device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")

    # Load data
    X, T, Y, done = load_data(args, False)
    X_test, T_test, Y_test, done_test = load_data(args, True)

    print(f"Data loaded: n_samples={len(X)}, n_features={X.shape[1]}")

    data_sizes = [1000, 500, 100]  # [1000, 500, 100]

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

        plt.figure(figsize=(18, 6))
        for idx, data_size in enumerate(data_sizes):
            train_idx = np.random.choice(len(X), size=data_size, replace=False)
            X_train = X[train_idx]
            T_train = T[train_idx]
            Y_train, Y_test_gt = Y[train_idx, i], Y_test[..., i]

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
                rmse = np.sqrt(np.mean((Y_test_hat - Y_test_gt) ** 2))
                print(f"Data size: {data_size}, Test RMSE: {rmse}")
            else:
                Y_test_pred = np.zeros_like(Y_test_hat)
                Y_test_pred[Y_test_hat > 0.5] = 1
                accuracy = np.mean((Y_test_pred == Y_test_gt))
                print(f"Data size: {data_size}, Test Accuracy: {accuracy}")

            plt.subplot(1, 3, idx + 1)
            plt.scatter(mu_0[T_test == 0], Y_test_gt[T_test == 0], alpha=0.1, label="T=0", color='tab:blue', s=1)
            plt.scatter(mu_1[T_test == 1], Y_test_gt[T_test == 1], alpha=0.1, label="T=1", color='tab:orange', s=1)
            mini, maxi = np.min(Y_test_gt), np.max(Y_test_gt)
            if i < 5: plt.plot([mini, maxi], [mini, maxi], 'k--', alpha=0.5)
            plt.xlabel(f"Predicted {name.replace('_', ' ')}", fontsize=18)
            plt.ylabel(f"True {name.replace('_', ' ')}", fontsize=18)
            left, right = plt.xlim()
            bottom, top = plt.ylim()
            mini = min(left, bottom)
            maxi = max(right, top)
            plt.xlim(mini, maxi)
            plt.ylim(mini, maxi)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.title(f"Data size: {data_size}, {'RMSE' if i < 5 else 'Accuracy'}: {rmse if i < 5 else accuracy:.8f}", fontsize=20)
            legend_dot0 = Line2D([0], [0], marker='o', color='tab:blue', linestyle='', markersize=6)
            legend_dot1 = Line2D([0], [0], marker='o', color='tab:orange', linestyle='', markersize=6)
            plt.legend([legend_dot0, legend_dot1], ["T=0", "T=1"], loc="upper left", fontsize=18)
 
        plt.tight_layout()
        plt.savefig(f"outputs/{args.env_id}/causalpfn_{name}_predictions_{args.data_type}_{args.reward_mapping}.png")