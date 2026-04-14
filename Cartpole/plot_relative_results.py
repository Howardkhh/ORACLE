import os
from typing import Literal
from collections import defaultdict
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import tyro

@dataclass
class Args:
    log_folder: str = "logs/"
    env_id: str = "cartpole"
    methods: tuple[Literal["dqn", "random", "mix"], ...] = ("mix",)
    method_ratio: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9,)
    oracle: tuple[Literal["both", "only", "no"], ...] = ("both", "no")
    oracle_source: tuple[Literal["simulator", "causalpfn"], ...] = ("causalpfn",)
    num_samples: tuple[int, ...] = (100, 200, 300, 400, 500, 600, 700, 800, 900, 1000)
    num_runs: int = 10

def ema(y, alpha=0.9):
    out = np.zeros_like(y, dtype=float)
    y[0]
    out[0] = y[0]
    for i in range(1, y.shape[0]):
        out[i] = alpha * out[i-1] + (1 - alpha) * y[i]
    return out

if __name__ == "__main__":
    args = tyro.cli(Args)
    os.makedirs("outputs/relative_training_results", exist_ok=True)

    reward_record = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))))
    pattern = re.compile(r"global step=(\d+), episodic_return=\[([\d.+-eE]+)\]")
    for method in args.methods:
        for ratio in args.method_ratio:
            for oracle in args.oracle:
                for source in args.oracle_source:
                    for num_sample in args.num_samples:
                        for run in range(1, args.num_runs+1):
                            log_path = f"{args.log_folder}/dqn_offline_{args.env_id}_{method}{ratio}_oracle_{oracle}_source_{source}_samples_{num_sample}_run_{run}.log"
                            # print(log_path)
                            with open(log_path, "r") as f:
                                has_data = False
                                for line in f:
                                    match = re.search(pattern, line)
                                    if match:
                                        has_data = True
                                        step = int(match.group(1))
                                        reward = float(match.group(2))
                                        reward_record[method][ratio][oracle][source][num_sample][step].append(reward)
                                    elif not line.startswith("Data loaded:"):
                                        print(line)
                            if not has_data:
                                print(f"No data found in {log_path}")
                    if oracle == "both":
                        for num_sample in args.num_samples:
                            if reward_record[method][ratio][oracle][source][num_sample*2].keys():
                                continue
                            for run in range(1, args.num_runs+1):
                                log_path = f"{args.log_folder}/dqn_offline_{args.env_id}_{method}{ratio}_oracle_{oracle}_source_{source}_samples_{num_sample*2}_run_{run}.log"
                                # print(log_path)
                                with open(log_path, "r") as f:
                                    has_data = False
                                    for line in f:
                                        match = re.search(pattern, line)
                                        if match:
                                            has_data = True
                                            step = int(match.group(1))
                                            reward = float(match.group(2))
                                            reward_record[method][ratio][oracle][source][num_sample*2][step].append(reward)
                                        elif not line.startswith("Data loaded:"):
                                            print(line)
                                if not has_data:
                                    print(f"No data found in {log_path}")

    # fig = plt.figure(figsize=(10, 10))
    fig = plt.figure(figsize=(16, 12))
    for ratio in args.method_ratio:
        # ax = fig.add_subplot(len(args.method_ratio), 1, int((ratio*10-1) / 2 + 1))
        ax = fig.add_subplot(1, 1, 1)
        method = "mix"
        oracle = "no"
        source = "causalpfn"

        final_steps = sorted(reward_record[method][ratio][oracle][source][args.num_samples[0]].keys())[-10:]
        records = np.array([[reward_record[method][ratio][oracle][source][num_sample][step] for step in final_steps] for num_sample in args.num_samples])
        final_rewards = records.mean(axis=1).mean(axis=1)
        plt.plot(args.num_samples, final_rewards, color="red", label="D_ref", linewidth=3)

        oracle = "both"
        final_steps = sorted(reward_record[method][ratio][oracle][source][args.num_samples[0]*2].keys())[-10:]
        test = [[reward_record[method][ratio][oracle][source][num_sample * 2][step] for step in final_steps] for num_sample in args.num_samples]
        for idx, tes in enumerate(test):
            for idx2, te in enumerate(tes):
                if len(te) != args.num_runs:
                    print("Missing data for ratio:", len(te), idx, idx2)
        records = np.array([[reward_record[method][ratio][oracle][source][num_sample * 2][step] for step in final_steps] for num_sample in args.num_samples])
        final_rewards = records.mean(axis=1).mean(axis=1)
        # plt.title(f"Random data ratio: {ratio*100:.0f}%, DQN expert data ratio: {(1 - ratio)*100:.0f}%")
        plt.title("CartPole-v1 Offline DQN Performance Comparison", fontsize=40)
        plt.plot(args.num_samples, final_rewards, color="blue", label="D_ref + D_causal", linewidth=3)
        plt.xticks(args.num_samples, fontsize=28)
        plt.yticks(range(0, 550, 100), fontsize=28)
        plt.xlabel("Number of Real Training Samples", fontsize=32)
        plt.ylabel("Total Return", fontsize=32)
        plt.ylim(0, 550)
        plt.grid()

        if ratio == args.method_ratio[0]:
            # plt.title(f"CartPole-v1 Offline DQN Relative Performance")
            plt.legend(loc="upper left", fontsize=28)
    plt.tight_layout()
    plt.savefig(f"outputs/relative_training_results/cartpole_offline_dqn.png")
    plt.close()