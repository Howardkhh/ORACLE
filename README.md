# Installation
First, install uv (https://docs.astral.sh/uv/#installation):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install matplotlib
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

### Help
Use `-h` or `--help` to see all options for each script. For example:

```bash
python collect_data.py -h
```

### Train DQN on Cartpole

```bash
cd cleanrl
python cleanrl/dqn.py --env-id CartPole-v1 --save-model

# replace your model path below
cd ../Cartpole
python collect_data.py --model ../cleanrl/runs/CartPole-v1__dqn__1__1762724960/dqn.cleanrl_model --method dqn

python collect_data.py --method random

python collect_data.py --model ../cleanrl/runs/CartPole-v1__dqn__1__1762724960/dqn.cleanrl_model --method dqn --is_validation_split

python collect_data.py --method random --is_validation_split

```

### Evaluate CausalPFN on Cartpole

```bash
python eval_causalpfn.py --data_type "mix"
```

### Train DQN with offline data and oracle data

```bash
python dqn_offline.py --env-id CartPole-v1 \
                      --reward_mapping angle \
                      --oracle both \
                      --oracle_source causalpfn \
                      --num_samples 500 \
                      --method_ratio 0.5 \
                      --seed 123 \
                      --exp_name exp_name
```

### Train DQN online with CausalPFN as environment

```bash
python dqn.py --env-id CausalPFNCartPole-v0 \
              --save_model
```
