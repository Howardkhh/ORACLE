#!/bin/bash
set -euo pipefail
trap "echo 'Interrupted! Killing all jobs...'; kill 0" SIGINT

num_samples=(200 400 600 800 1000)
method_ratios=(0.0 0.2 0.4 0.6 0.8 1.0)
oracles=("no" "only") # ("both" "no" "only")
s="simulator" # source of oracle
runs=5
num_parallel=30

mkdir -p logs

if command -v nvidia-smi &>/dev/null; then
    NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
else
    NUM_GPUS=0
fi

i=0
for n in "${num_samples[@]}"; do
  for mr in "${method_ratios[@]}"; do
    for o in "${oracles[@]}"; do
      for r in $(seq 1 $runs); do
          exp_name="dqn_offline_cartpole_mix${mr}_oracle_${o}_source_${s}_samples_${n}_run_${r}"
          if [ -s "logs/${exp_name}.log" ]; then
            echo "Experiment $exp_name already completed. Skipping."
            continue
          fi
          ((i=i+1))
          gpu_id=$(( (i - 1) % NUM_GPUS ))
          CUDA_VISIBLE_DEVICES=${gpu_id}
          echo "Starting experiment: $exp_name"
          python dqn_offline.py --env-id CartPole-v1 \
                                --reward_mapping angle \
                                --oracle "$o" \
                                --oracle_source "$s" \
                                --num_samples "$n" \
                                --method_ratio "$mr" \
                                --seed "$r" \
                                --exp_name "$exp_name" > "logs/${exp_name}.log" &
          if (( i % $num_parallel == 0 )); then
              wait
          fi
      done
    done
  done
done

num_samples=(400 800 1200 1600 2000)
i=0
for n in "${num_samples[@]}"; do
  for mr in "${method_ratios[@]}"; do
    o="both"
    for r in $(seq 1 $runs); do
      exp_name="dqn_offline_cartpole_mix${mr}_oracle_${o}_source_${s}_samples_${n}_run_${r}"
      if [ -s "logs/${exp_name}.log" ]; then
        echo "Experiment $exp_name already completed. Skipping."
        continue
      fi
      ((i=i+1))
      gpu_id=$(( (i - 1) % NUM_GPUS ))
      CUDA_VISIBLE_DEVICES=${gpu_id}
      echo "Starting experiment: $exp_name"
      python dqn_offline.py --env-id CartPole-v1 \
                            --reward_mapping angle \
                            --oracle "$o" \
                            --oracle_source "$s" \
                            --num_samples "$n" \
                            --method_ratio "$mr" \
                            --seed "$r" \
                            --exp_name "$exp_name" > "logs/${exp_name}.log" &
      if (( i % $num_parallel == 0 )); then
          wait
      fi
    done
  done
done

wait