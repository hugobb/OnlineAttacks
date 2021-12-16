
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#!/bin/bash

NAME="non_robust_final/imagenet_pgd_eps_4_255/"
SLURM="/private/home/joeybose/OnlineAttacks/configs/learnfair.yml"
NUM_RUNS="10"
SAME_MODEL_TYPE=""
K="2 10 100 1000"

# MNIST
#DATASET="mnist"
#python -m online_attacks.scripts.random_eval --k $K --dataset $DATASET --attacker_type none --name $NAME/none --slurm $SLURM --num_runs $NUM_RUNS $SAME_MODEL_TYPE
#python -m online_attacks.scripts.random_eval --k $K --dataset $DATASET --attacker_type fgsm --name $NAME/fgsm --slurm $SLURM --num_runs $NUM_RUNS $SAME_MODEL_TYPE
#python -m online_attacks.scripts.random_eval --k $K --dataset $DATASET --attacker_type pgd --name $NAME/pgd --slurm $SLURM --num_runs $NUM_RUNS $SAME_MODEL_TYPE
#python -m online_attacks.scripts.random_eval --k $K --dataset $DATASET --attacker_type cw --name $NAME/cw --slurm $SLURM --num_runs $NUM_RUNS $SAME_MODEL_TYPE

# CIFAR
#DATASET="cifar"
#BATCH_SIZE="128"
#python -m online_attacks.scripts.random_eval --k $K --dataset $DATASET --attacker_type none --name $NAME/none --slurm $SLURM --batch_size $BATCH_SIZE --num_runs $NUM_RUNS $SAME_MODEL_TYPE
#python -m online_attacks.scripts.random_eval --k $K --dataset $DATASET --attacker_type fgsm --name $NAME/fgsm --slurm $SLURM --batch_size $BATCH_SIZE --num_runs $NUM_RUNS $SAME_MODEL_TYPE
#python -m online_attacks.scripts.random_eval --k $K --dataset $DATASET --attacker_type pgd --name $NAME/pgd --slurm $SLURM --batch_size $BATCH_SIZE --num_runs $NUM_RUNS $SAME_MODEL_TYPE
#python -m online_attacks.scripts.random_eval --k $K --dataset $DATASET --attacker_type cw --name $NAME/cw --slurm $SLURM --batch_size $BATCH_SIZE --num_runs $NUM_RUNS $SAME_MODEL_TYPE

# IMAGENET
DATASET="imagenet"
BATCH_SIZE="32"
#python -m online_attacks.scripts.random_eval --k $K --dataset $DATASET --attacker_type none --name $NAME/none --slurm $SLURM --batch_size $BATCH_SIZE --num_runs $NUM_RUNS $SAME_MODEL_TYPE
#python -m online_attacks.scripts.random_eval --k $K --dataset $DATASET --attacker_type fgsm --name $NAME/fgsm --slurm $SLURM --batch_size $BATCH_SIZE --num_runs $NUM_RUNS $SAME_MODEL_TYPE &
python -m online_attacks.scripts.random_eval --k $K --dataset $DATASET --attacker_type pgd --name $NAME/pgd --slurm $SLURM --batch_size $BATCH_SIZE --num_runs $NUM_RUNS $SAME_MODEL_TYPE &
#python -m online_attacks.scripts.random_eval --k $K --dataset $DATASET --attacker_type cw --name $NAME/cw --slurm $SLURM --batch_size $BATCH_SIZE --num_runs $NUM_RUNS $SAME_MODEL_TYPE
