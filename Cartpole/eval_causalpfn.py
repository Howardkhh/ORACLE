import os
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import tyro
from causalpfn import CATEEstimator
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm

@dataclass
class Args:
    data_type: Literal["random", "dqn", "mix"]
    data_folder: str = "data/"
    env_id: str = "CartPole-v1"
    observation_mode: Literal["state", "image", "feature"] = "state"
    stack_frames: int = 1
    cuda: bool = True

def load_data(args: Args, validation: bool):
    if args.data_type == "random" or args.data_type == "dqn":
        with np.load(f"{args.data_folder}/{args.env_id}/{args.data_type}_{args.observation_mode}_stack{args.stack_frames}{'_validation' if validation else ''}.npz") as data:
            X = data["states"].astype(np.float32)
            T = data["actions"].astype(np.float32)
            if args.observation_mode == "feature":
                X = X.reshape(X.shape[0], -1)
                next_states = data["next_states"][:, :, -1].astype(np.float32) # get last frame only
            Y = np.concatenate([data["rewards"].astype(np.float32)[..., np.newaxis], next_states, data["dones"].astype(np.float32)[..., np.newaxis]], axis=-1)
            done = data["dones"]
    elif args.data_type == "mix":
        with np.load(f"{args.data_folder}/{args.env_id}/random_{args.observation_mode}_stack{args.stack_frames}{'_validation' if validation else ''}.npz") as data:
            X_random = data["states"].astype(np.float32)
            T_random = data["actions"].astype(np.float32)
            if args.observation_mode == "feature":
                X = X.reshape(X.shape[0], -1)
                next_states = data["next_states"][:, :, -1].astype(np.float32) # get last frame only
            Y_random = np.concatenate([data["rewards"].astype(np.float32)[..., np.newaxis], next_states, data["dones"].astype(np.float32)[..., np.newaxis]], axis=-1)
            done_random = data["dones"]
        with np.load(f"{args.data_folder}/{args.env_id}/dqn_{args.observation_mode}_stack{args.stack_frames}{'_validation' if validation else ''}.npz") as data:
            X_dqn = data["states"].astype(np.float32)
            T_dqn = data["actions"].astype(np.float32)
            if args.observation_mode == "feature":
                X = X.reshape(X.shape[0], -1)
                next_states = data["next_states"][:, :, -1].astype(np.float32) # get last frame only
            Y_dqn = np.concatenate([data["rewards"].astype(np.float32)[..., np.newaxis], next_states, data["dones"].astype(np.float32)[..., np.newaxis]], axis=-1)
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

def eval_state(train, test, data_sizes=[1000, 500, 100], device="cuda"):
    X, T, Y = train
    X_test, T_test, Y_test = test
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


def eval_feature(train, test, data_sizes=[1000, 500, 100], device="cuda"):
    X, T, Y = train
    X_test, T_test, Y_test = test
    target_names = ["Reward"] + [f"Feature_{i}" for i in range(Y.shape[2]-2)] + ["Done"]
    for idx, data_size in enumerate(data_sizes):
        print(f"\nEvaluating CATE estimation with data size: {data_size}")
        feature_rmse = 0
        for i, name in enumerate(tqdm(target_names)):
            train_idx = np.random.choice(len(X), size=data_size, replace=False)
            X_train = X[train_idx]
            T_train = T[train_idx]
            Y_train, Y_test_gt = Y[train_idx, T_train.astype(int), i], Y_test[np.arange(len(Y_test)), T_test.astype(int), i]
            causalpfn_cate = CATEEstimator(
                model_path="/home/howardkhh/.cache/causalpfn/models--vdblm--causalpfn/snapshots/ccfc5083f28270d09356b8c35190073df17798d5/causalpfn_v0.pt",
                device=device,
                verbose=False,
            )
            causalpfn_cate.fit(X_train, T_train, Y_train)
            cate_hat, mu_0, mu_1 = causalpfn_cate.estimate_cate(X_test)
            Y_test_hat = mu_0 * (1 - T_test) + mu_1 * T_test

            if i == 0: # done
                rmse = np.sqrt(np.mean((Y_test_hat - Y_test_gt) ** 2))
                print(f"Data size: {data_size}, Target: {name}, Test RMSE: {rmse}")
            elif i < len(target_names) - 1:
                feature_rmse += np.sqrt(np.mean((Y_test_hat - Y_test_gt) ** 2))
            else:
                Y_test_pred = np.zeros_like(Y_test_hat)
                Y_test_pred[Y_test_hat > 0.5] = 1
                accuracy = np.mean((Y_test_pred == Y_test_gt))
                print(f"Data size: {data_size}, Target: {name}, Test Accuracy: {accuracy}")
            
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
            plt.title(f"Data size: {data_size}, {'RMSE' if i < len(target_names) - 1 else 'Accuracy'}: {rmse if i < len(target_names) - 1 else accuracy:.8f}", fontsize=20)
            legend_dot0 = Line2D([0], [0], marker='o', color='tab:blue', linestyle='', markersize=6)
            legend_dot1 = Line2D([0], [0], marker='o', color='tab:orange', linestyle='', markersize=6)
            plt.legend([legend_dot0, legend_dot1], ["T=0", "T=1"], loc="upper left", fontsize=18)

        print(f"Average feature RMSE: {feature_rmse / (len(target_names) - 2)}")

if __name__ == "__main__":
    args = tyro.cli(Args)

    device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")

    # Load data
    X, T, Y, done = load_data(args, False)
    X_test, T_test, Y_test, done_test = load_data(args, True)

    print(f"Data loaded: n_samples={len(X)}, n_features={X.shape[1]}")

    data_sizes = [1000, 500, 100]  # [1000, 500, 100]

    if args.observation_mode == "state":
        eval_state((X, T, Y), (X_test, T_test, Y_test), data_sizes=data_sizes, device=device)
    elif args.observation_mode == "feature":
        eval_feature((X, T, Y), (X_test, T_test, Y_test), data_sizes=data_sizes, device=device)
    else:
        raise NotImplementedError()

    