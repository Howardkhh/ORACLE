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

sns.set(style="whitegrid")

@dataclass
class Args:
    log_folder: str = "logs/"
    env_id: str = "cartpole"
    method_ratio: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0) # (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    oracle: tuple[Literal["both", "only", "no"], ...] = ("both", "no", "only")
    oracle_source: tuple[Literal["simulator", "causalpfn"], ...] = ("simulator", "causalpfn")
    num_samples: tuple[int, ...] = (200, 400, 600, 800, 1000) # (100, 200, 300, 400, 500, 600, 700, 800, 900, 1000)
    num_runs: int = 5

def ema(y, alpha=0.9):
    out = np.zeros_like(y, dtype=float)
    y[0]
    out[0] = y[0]
    for i in range(1, y.shape[0]):
        out[i] = alpha * out[i-1] + (1 - alpha) * y[i]
    return out

if __name__ == "__main__":
    args = tyro.cli(Args)
    os.makedirs("outputs/training_results", exist_ok=True)

    reward_record = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))))
    pattern = re.compile(r"global step=(\d+), episodic_return=\[([\d.+-eE]+)\]")
    for ratio in args.method_ratio:
        for oracle in args.oracle:
            for source in args.oracle_source:
                if oracle != "both":
                    for num_sample in args.num_samples:
                        for run in range(1, args.num_runs+1):
                            log_path = f"{args.log_folder}/dqn_offline_{args.env_id}_mix{ratio}_oracle_{oracle}_source_{source}_samples_{num_sample}_run_{run}.log"
                            # print(log_path)
                            with open(log_path, "r") as f:
                                has_data = False
                                for line in f:
                                    match = re.search(pattern, line)
                                    if match:
                                        has_data = True
                                        step = int(match.group(1))
                                        reward = float(match.group(2))
                                        reward_record[ratio][oracle][source][num_sample][step].append(reward)
                                    elif not line.startswith("Data loaded:"):
                                        print(line)
                            if not has_data:
                                print(f"No data found in {log_path}")
                else: # oracle == "both"
                    for num_sample in args.num_samples:
                        if reward_record[ratio][oracle][source][num_sample*2].keys():
                            continue
                        for run in range(1, args.num_runs+1):
                            log_path = f"{args.log_folder}/dqn_offline_{args.env_id}_mix{ratio}_oracle_{oracle}_source_{source}_samples_{num_sample*2}_run_{run}.log"
                            # print(log_path)
                            with open(log_path, "r") as f:
                                has_data = False
                                for line in f:
                                    match = re.search(pattern, line)
                                    if match:
                                        has_data = True
                                        step = int(match.group(1))
                                        reward = float(match.group(2))
                                        reward_record[ratio][oracle][source][num_sample*2][step].append(reward)
                                    elif not line.startswith("Data loaded:"):
                                        print(line)
                            if not has_data:
                                print(f"No data found in {log_path}")

    for num_sample in args.num_samples:
        for ratio in args.method_ratio:
            plt.figure(figsize=(10, 6))
            for oracle in args.oracle:
                if oracle == "both":
                    num_sample_used = num_sample * 2
                else:
                    num_sample_used = num_sample
                for source in args.oracle_source:
                    records = reward_record[ratio][oracle][source][num_sample_used]
                    steps = sorted(records.keys())
                    # (num_steps, num_runs)
                    # print(f"{ratio}, {oracle}, {source}, {num_sample_used}")
                    rewards = ema(np.array([records[step][:args.num_runs] for step in steps]))
                    # rewards = np.array([np.mean(records[step]) for step in steps])
                    # rewards_smooth = ema(rewards)
                    # rewards_smooth_std = np.array([np.std(records[step]) for step in steps])
                    # rewards_smooth_error = rewards_smooth_std / np.sqrt(args.num_runs)
                    df = pd.DataFrame({
                        "Step": np.repeat(steps, args.num_runs),
                        "Total Reward": rewards.flatten()
                    })

                    if oracle == "no":
                        color = "green"
                    elif oracle == "only":
                        color = "red"
                    elif oracle == "both":
                        color = "blue"
                        # records_double_data = reward_record[ratio][oracle][source][num_sample * 2]
                        # steps_double_data = sorted(records_double_data.keys())
                        # rewards_double_data = ema(np.array([records_double_data[step][:args.num_runs] for step in steps_double_data]))
                        # df_double_data = pd.DataFrame({
                        #     "Step": np.repeat(steps_double_data, args.num_runs),
                        #     "Total Reward": rewards_double_data.flatten()
                        # })
                        # rewards_double_data_smooth = ema(np.array([np.mean(records_double_data[step]) for step in steps_double_data]))
                        # rewards_double_smooth_std = np.array([np.std(records_double_data[step]) for step in steps_double_data])
                        # rewards_double_data_smooth_error = rewards_double_smooth_std / np.sqrt(args.num_runs)
                        # plt.plot(steps_double_data, rewards_double_data_smooth, label=f"{method} - oracle: {oracle} (double data)", color="blue", linestyle="--")
                        # plt.fill_between(steps_double_data,
                        #                 rewards_double_data_smooth - rewards_double_data_smooth_error * 1.96,
                        #                 rewards_double_data_smooth + rewards_double_data_smooth_error * 1.96,
                        #                 alpha=0.2, color="blue")
                        # sns.lineplot(
                        #     data=df_double_data,
                        #     x="Step",
                        #     y="Total Reward",
                        #     label=f"Random {ratio} / Expert DQN {1-ratio} - oracle: {oracle} (double data)",
                        #     color=color,
                        #     linestyle="--",
                        #     errorbar="sd"
                        # )
                    # plt.plot(steps, rewards_smooth, label=f"{method} - oracle: {oracle}", color=color)
                    # plt.fill_between(steps,
                    #                 rewards_smooth - rewards_smooth_error * 1.96,
                    #                 rewards_smooth + rewards_smooth_error * 1.96,
                    #                 alpha=0.2, color=color)
                    sns.lineplot(
                        data=df,
                        x="Step",
                        y="Total Reward",
                        label=f"Random {ratio} / Expert DQN {1-ratio} - oracle: {oracle}",
                        color=color,
                        errorbar="sd"
                    )
            plt.title(f"CartPole-v1 Offline DQN Learning Curve (data=Mixed Random {ratio} / Expert DQN {1-ratio}, oracle source={source}, num_samples={num_sample})")
            plt.xlabel("Steps")
            plt.ylabel("Episodic Return")
            plt.ylim(0, 550)
            plt.legend(loc="upper left")
            plt.grid()
            plt.savefig(f"outputs/training_results/cartpole_offline_dqn_method_mix{ratio}_oracle_source_{source}_num_samples_{num_sample}.png")
            plt.close()