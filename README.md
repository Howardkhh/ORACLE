# Installation
First, install uv (https://docs.astral.sh/uv/#installation):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.10 .venv
source .venv/bin/activate
```

Install CleanRL:

```bash
uv pip install -e cleanrl
```

Install CuasalPFN:

```bash
uv pip install -e CausalPFN
```

# Usage

### Download real and oracle data

Download from [here](https://drive.google.com/file/d/1CaZDCUpWNVOF34nXT6P9VM59sCKLZVhT/view?usp=sharing) and unzip the data into `Cartpole/`.

Note the oracle data has not been tested. There might be issue with them. Use with caution.

```bash
Cartpole
└── data
    └── Cartpole-v1
        ├── dqn.npz # real data collected by trained DQN policy, note the reward is original reward (0 or 1)
        ├── dqn_oracle_remapped.npz # oracle data collected by trained DQN policy, reward = cos(angle)
        ├── random.npz # real data collected by random policy
        └── random_oracle_remapped.npz # oracle data collected by random policy
```

### Train DQN on Cartpole

```bash
cd cleanrl
python cleanrl/dqn.py --env-id CartPole-v1 --save-model

# replace your model path below
python collect_data.py --model ../cleanrl/runs/CartPole-v1__dqn__1__1762724960/dqn.cleanrl_model
```

### Evaluate CausalPFN on Cartpole

```bash
cd CausalPFN
python eval_causalpfn.py --data_type ["dqn", "random", "mix"] --reward_mapping: ["original", "angle"]
```

### Generate Oracle Data

```bash
cd CausalPFN
python generate_oracle.py --reward_mapping ["original", "angle"]
```

### Train DQN with offline data and oracle data

```bash
```