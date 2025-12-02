import os
import random
from dataclasses import dataclass
from typing import Callable, Literal

import gymnasium as gym
import numpy as np
import torch
import tyro

from cleanrl.dqn import QNetwork

@dataclass
class Args:
    method: Literal["dqn", "random"] = "dqn"
    env_id: str = "CartPole-v1"
    total_steps: int = 1000
    max_steps_per_episode: int = 100
    seed: int = 1
    model: str = ""
    cuda: bool = True
    is_validation_split: bool = False

def make_env(env_id, seed):
    def thunk():
        env = gym.make(env_id)
        env.action_space.seed(seed)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        return env

    return thunk

def collect(
    model_path: str,
    make_env: Callable,
    env_id: str,
    total_steps: int,
    max_steps_per_episode: int,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    epsilon: float = 0.0,
):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0)])
    if epsilon < 1.0:
        model = Model(envs).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()

    obs, _ = envs.reset()
    rec_states, rec_next_states, rec_actions, rec_rewards, rec_dones = np.ndarray((total_steps, obs.shape[1])), np.ndarray((total_steps, obs.shape[1])), np.ndarray((total_steps,)), np.ndarray((total_steps,)), np.ndarray((total_steps,))
    step = 0
    step_in_episode = 0
    while step < total_steps:
        step += 1
        step_in_episode += 1
        if epsilon == 1.0 or random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = model(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        done = np.logical_or(terminations, truncations)
        rec_states[step - 1] = obs[0]
        rec_next_states[step - 1] = next_obs[0]
        rec_actions[step - 1] = actions[0]
        rec_rewards[step - 1] = rewards[0]
        rec_dones[step - 1] = done[0]
        if "final_info" in infos:
            done = True
            rec_next_states[step - 1] = infos["final_observation"][0]
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(f"eval_episode={step}, episodic_return={info['episode']['r']}")
        if "final_info" in infos or step_in_episode >= max_steps_per_episode:
            obs, _ = envs.reset()
            step_in_episode = 0
            continue

        obs = next_obs

    return rec_states, rec_next_states, rec_actions, rec_rewards, rec_dones

if __name__ == "__main__":
    args = tyro.cli(Args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")


    states, next_states, actions, rewards, dones = collect(
        args.model,
        make_env,
        args.env_id,
        args.total_steps,
        args.max_steps_per_episode,
        QNetwork,
        device=device,
        epsilon=0.0 if args.method == "dqn" else 1.0,
    )

    os.makedirs(f"data/{args.env_id}", exist_ok=True)
    np.savez_compressed(
        f"data/{args.env_id}/{args.method}{'_validation' if args.is_validation_split else ''}.npz",
        states=states,
        next_states=next_states,
        actions=actions,
        rewards=rewards,
        dones=dones,
    )
    np.savez_compressed(
        f"data/{args.env_id}/{args.method}_remapped{'_validation' if args.is_validation_split else ''}.npz",
        states=states,
        next_states=next_states,
        actions=actions,
        rewards=np.cos(next_states[:, 2]),
        dones=dones,
    )