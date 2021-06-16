#!/bin/bash
python online_attacks/experiments/stochastic_toy.py --online_params.online_type stochastic_optimistic --wandb --exhaust --online_params.N 100 --max_perms 10000 --K 20 --eps 1.0  --namestr Exhaust-Var1-Toy-Stochastic-Optimistic
python online_attacks/experiments/stochastic_toy.py --online_params.online_type stochastic_optimistic --wandb --exhaust --online_params.N 100 --max_perms 10000 --K 20 --eps 5.0  --namestr Exhaust-Var5-Toy-Stochastic-Optimistic
python online_attacks/experiments/stochastic_toy.py --online_params.online_type stochastic_optimistic --wandb --exhaust --online_params.N 100 --max_perms 10000 --K 20 --eps 10.0  --namestr Exhaust-Var10-Toy-Stochastic-Optimistic

# Virtual
python online_attacks/experiments/stochastic_toy.py --online_params.online_type stochastic_virtual --wandb --exhaust --online_params.N 100 --max_perms 10000 --K 20 --eps 1.0  --namestr Exhaust-Var1-Toy-Stochastic-Virtual
python online_attacks/experiments/stochastic_toy.py --online_params.online_type stochastic_virtual --wandb --exhaust --online_params.N 100 --max_perms 10000 --K 20 --eps 5.0  --namestr Exhaust-Var5-Toy-Stochastic-Virtual
python online_attacks/experiments/stochastic_toy.py --online_params.online_type stochastic_virtual --wandb --exhaust --online_params.N 100 --max_perms 10000 --K 20 --eps 10.0  --namestr Exhaust-Var10-Toy-Stochastic-Virtual

# Virtual Plus
python online_attacks/experiments/stochastic_toy.py --online_params.online_type stochastic_modified_virtual --wandb --exhaust --online_params.N 100 --max_perms 10000 --K 20 --eps 1.0  --namestr Exhaust-Var1-Toy-Stochastic-Virtual-Plus
python online_attacks/experiments/stochastic_toy.py --online_params.online_type stochastic_modified_virtual --wandb --exhaust --online_params.N 100 --max_perms 10000 --K 20 --eps 5.0  --namestr Exhaust-Var5-Toy-Stochastic-Virtual-Plus
python online_attacks/experiments/stochastic_toy.py --online_params.online_type stochastic_modified_virtual --wandb --exhaust --online_params.N 100 --max_perms 10000 --K 20 --eps 10.0  --namestr Exhaust-Var10-Toy-Stochastic-Virtual-Plus

# Single-Ref
python online_attacks/experiments/stochastic_toy.py --online_params.online_type stochastic_single_ref --wandb --exhaust --online_params.N 100 --max_perms 10000 --K 20 --eps 1.0  --namestr Exhaust-Var1-Toy-Stochastic-Single-Ref
python online_attacks/experiments/stochastic_toy.py --online_params.online_type stochastic_single_ref --wandb --exhaust --online_params.N 100 --max_perms 10000 --K 20 --eps 5.0  --namestr Exhaust-Var5-Toy-Stochastic-Single-Ref
python online_attacks/experiments/stochastic_toy.py --online_params.online_type stochastic_single_ref --wandb --exhaust --online_params.N 100 --max_perms 10000 --K 20 --eps 10.0  --namestr Exhaust-Var10-Toy-Stochastic-Single-Ref
