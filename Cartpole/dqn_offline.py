# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import os
import random
import time
from dataclasses import dataclass
from typing import Literal

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.buffers import ReplayBuffer
from dqn import QNetwork, make_env
from dqn_offline_utils import *

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = ""
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 10000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 500
    """the timesteps it takes to update the target network"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    learning_starts: int = 10000
    """timestep to start learning"""
    train_frequency: int = 10
    """the frequency of training"""
    evaluation_frequency: int = 50000
    """the timesteps it takes to evaluate the agent"""

    # data arguments
    data_folder: str = "data/"
    reward_mapping: Literal["original", "angle"] = "angle"
    method_ratio: float = 0.5
    """ratio of random data in the mix"""
    oracle: Literal["only", "both", "no"] = "no" # only: oracle only; both: oracle + offline; no: offline only
    oracle_source: Literal["simulator", "causalpfn"] = "causalpfn"
    num_samples: int = 100



if __name__ == "__main__":
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Load data
    sample_idx = None
    if args.oracle in ["only", "both"]:
        states_oracle, actions_oracle, rewards_oracle, next_states_oracle, dones_oracle, sample_idx = load_data(args, True)
    if args.oracle in ["no", "both"]:
        states_real, actions_real, rewards_real, next_states_real, dones_real, _ = load_data(args, False, sample_idx)

    if args.oracle == "both":
        states = np.stack([states_oracle, states_real], axis=0)
        actions = np.stack([actions_oracle, actions_real], axis=0)
        rewards = np.stack([rewards_oracle, rewards_real], axis=0)
        next_states = np.stack([next_states_oracle, next_states_real], axis=0)
        dones = np.stack([dones_oracle, dones_real], axis=0)
        data_size = states.shape[1] * 2
        assert args.num_samples == data_size, f"num_samples not equal to dataset size: {args.num_samples} != {data_size}"
    elif args.oracle == "only":
        states, actions, rewards, next_states, dones = states_oracle, actions_oracle, rewards_oracle, next_states_oracle, dones_oracle
        data_size = states.shape[0]
        assert args.num_samples == data_size, f"num_samples not equal to dataset size: {args.num_samples} != {data_size}"
    elif args.oracle == "no":
        states, actions, rewards, next_states, dones = states_real, actions_real, rewards_real, next_states_real, dones_real
        data_size = states.shape[0]
        assert args.num_samples == data_size, f"num_samples not equal to dataset size: {args.num_samples} != {data_size}"
    
    print(f"Data loaded: n_samples={data_size}", flush=True)

    if args.num_samples > (args.batch_size * args.total_timesteps - args.learning_starts) // args.train_frequency:
        print(f"Warning: dataset size ({args.num_samples}) is larger than total training steps * batch size ({(args.batch_size * args.total_timesteps - args.learning_starts) // args.train_frequency}). Some data will not be used during training.", flush=True)
    if args.buffer_size < args.num_samples:
        print(f"Warning: buffer size ({args.buffer_size}) is smaller than dataset size ({args.num_samples}). Expanding buffer size to {args.num_samples}.", flush=True)
        args.buffer_size = args.num_samples

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )

    if args.oracle == "both":
        for idx in range(states.shape[1]):
            assert (states[0][idx] == states[1][idx]).all(), "Mismatch between oracle and real data states"
            assert (actions[0][idx] + actions[1][idx] == 1).all(), "Mismatch between oracle and real data actions"
            rb.add(states[0][idx], next_states[0][idx], actions[0][idx], rewards[0][idx], dones[0][idx], [])
            rb.add(states[1][idx], next_states[1][idx], actions[1][idx], rewards[1][idx], dones[1][idx], [])
    else:
        for idx in range(states.shape[0]):
            rb.add(states[idx], next_states[idx], actions[idx], rewards[idx], dones[idx], [])

    start_time = time.time()

    # obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        # epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        # if random.random() < epsilon:
        #     actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        # else:
        #     q_values = q_network(torch.Tensor(obs).to(device))
        #     actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        # next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        # if "final_info" in infos:
        #     for info in infos["final_info"]:
        #         if info and "episode" in info:
        #             print(f"global_step={global_step}, episodic_return={info['episode']['r']}", flush=True)
        #             writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
        #             writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        # real_next_obs = next_obs.copy()
        # for idx, trunc in enumerate(truncations):
        #     if trunc:
        #         real_next_obs[idx] = infos["final_observation"][idx]
        # rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        # obs = next_obs

        
        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations).max(dim=1)
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    # print("SPS:", int(global_step / (time.time() - start_time)), flush=True)
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )

        if global_step % args.evaluation_frequency == 0:
                obs, _ = envs.reset()
                done = False
                while not done:
                    q_values = q_network(torch.Tensor(obs).to(device))
                    actions = torch.argmax(q_values, dim=1).cpu().numpy()
                    next_obs, _, _, _, infos = envs.step(actions)
                    if "final_info" in infos:
                        for info in infos["final_info"]:
                            if "episode" not in info:
                                continue
                            print(f"global step={global_step}, episodic_return={info['episode']['r']}", flush=True)
                        break
                    obs = next_obs

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}", flush=True)
        from cleanrl_utils.evals.dqn_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=QNetwork,
            device=device,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "DQN", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()
