import os
import random
from dataclasses import dataclass
from typing import Callable, Literal
from copy import deepcopy

import gymnasium as gym
import numpy as np
import torch
import tyro

from cleanrl.dqn import QNetwork
from cleanrl_utils.custom_wrappers import DINOv2FeatureWrapper, RenderObservation, PreprocessObservation

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
    observation_mode: Literal["state", "image", "feature"] = "state"
    stack_frames: int = 1

def make_env(env_id, seed, stack_frames=1, observation_mode='state'):
    def thunk():
        render_mode = "rgb_array" if observation_mode in ['image', 'feature'] else None
        env = gym.make(env_id, render_mode=render_mode)

        if observation_mode == 'image' or observation_mode == 'feature':
            env = RenderObservation(env)

        if observation_mode == 'image':
            env = gym.wrappers.GrayScaleObservation(env, keep_dim=True)
            env = PreprocessObservation(env)
        if observation_mode == 'feature':
            env = DINOv2FeatureWrapper(env)

        if stack_frames > 1:
            env = gym.wrappers.FrameStack(env, stack_frames)

        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)

        return env

    return thunk

def collect(
    args: Args,
    model_path: str,
    make_env: Callable,
    env_id: str,
    total_steps: int,
    max_steps_per_episode: int,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    epsilon: float = 0.0,
):
    envs = [make_env(env_id, 0, args.stack_frames, args.observation_mode)() for _ in range(2)]
    if epsilon < 1.0:
        model = Model(envs[0], args.observation_mode).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()

    obs, _ = envs[0].reset(seed=args.seed)
    envs[1].reset(seed=args.seed)
    rec_states, rec_next_states, rec_actions, rec_rewards, rec_dones = np.ndarray((total_steps, *obs.shape)), np.ndarray((total_steps, 2, *obs.shape)), np.ndarray((total_steps,)), np.ndarray((total_steps, 2)), np.ndarray((total_steps, 2))
    step = 0
    step_in_episode = 0
    while step < total_steps:
        step += 1
        step_in_episode += 1
        if epsilon == 1.0 or random.random() < epsilon:
            actions = np.array([envs[0].action_space.sample() for _ in range(envs[0].num_envs)])
        else:
            q_values = model(torch.Tensor(np.array(obs)).to(device))
            actions = torch.argmax(q_values).cpu().numpy()
        envs_0 = envs[0]
        envs_1 = envs[1]
        actions_0, actions_1 = 0, 1
        next_obs_0, _, terminations_0, truncations_0, infos_0 = envs_0.step(actions_0)
        next_obs_1, _, terminations_1, truncations_1, infos_1 = envs_1.step(actions_1)
        rewards_0, rewards_1 = np.cos(envs_0.unwrapped.state[2]), np.cos(envs_1.unwrapped.state[2])
        done_0 = np.logical_or(terminations_0, truncations_0)
        done_1 = np.logical_or(terminations_1, truncations_1)
        rec_states[step - 1] = obs
        rec_next_states[step - 1] = np.stack([next_obs_0, next_obs_1]) # next state of both actions
        rec_actions[step - 1] = actions # chosen action
        rec_rewards[step - 1] = np.stack([rewards_0, rewards_1]) # reward of both actions
        rec_dones[step - 1] = np.stack([done_0, done_1]) # done of both actions

        if actions == 0:
            envs[1].unwrapped.state = deepcopy(envs[0].unwrapped.state)
            next_obs = next_obs_0
            infos = infos_0
        else:
            envs[0].unwrapped.state = deepcopy(envs[1].unwrapped.state)
            next_obs = next_obs_1
            infos = infos_1
        if "episode" in infos:
            print(f"eval_episode={step}, episodic_return={infos['episode']['r']}")
        if "episode" in infos or step_in_episode >= max_steps_per_episode:
            obs, _ = envs[0].reset(seed=args.seed + step)
            envs[1].reset(seed=args.seed + step)
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
        args,
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
        f"data/{args.env_id}/{args.method}_{args.observation_mode}_stack{args.stack_frames}{'_validation' if args.is_validation_split else ''}.npz",
        states=states,
        next_states=next_states,
        actions=actions,
        rewards=rewards,
        dones=dones,
    )
    # np.savez_compressed(
    #     f"data/{args.env_id}/{args.method}_remapped{'_validation' if args.is_validation_split else ''}.npz",
    #     states=states,
    #     next_states=next_states,
    #     actions=actions,
    #     rewards=np.cos(next_states[:, 2]),
    #     dones=dones,
    # )