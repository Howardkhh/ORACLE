from cleanrl_utils.evals.dqn_eval import evaluate
from dataclasses import dataclass

import tyro
from cleanrl.dqn import QNetwork
import gymnasium as gym
import torch
import time
from causalpfn_env import *

@dataclass
class Args:
    env_id: str = "CausalPFNCartPole-v0" # "CartPole-v1"
    exp_name: str = "causalpfn_cartpole_dqn"
    seed: int = 1
    model_path: str = ""

def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}", episode_trigger=lambda x: True)
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)

        return env

    return thunk

if __name__ == "__main__":

    args = tyro.cli(Args)

    if args.env_id == "CausalPFNCartPole-v0":
        gym.register(
            id="CausalPFNCartPole-v0",
            entry_point=CausalPFNCartPoleEnv,
            vector_entry_point=CausalPFNCartPoleVectorEnv,
            max_episode_steps=500,
        )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    episodic_returns = evaluate(
        args.model_path,
        make_env,
        args.env_id,
        eval_episodes=10,
        run_name=f"{run_name}-eval",
        Model=QNetwork,
        device=device,
        epsilon=0.05,
    )