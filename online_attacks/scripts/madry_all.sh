#!/bin/bash

NAME="madry-icml"
SLURM="configs/learnfair.yaml"
NUM_RUNS="1000"

# MNIST
DATASET="mnist"
#python -m online_attacks.scripts.online_attacks_sweep --dataset $DATASET --model_type madry --model_name adv_trained --attacker_type none --name $NAME/mnist/none --num_runs $NUM_RUNS --slurm $SLURM
#python -m online_attacks.scripts.online_attacks_sweep --dataset $DATASET --model_type madry --model_name adv_trained --attacker_type fgsm --name $NAME/mnist/fgsm --num_runs $NUM_RUNS --slurm $SLURM
#python -m online_attacks.scripts.online_attacks_sweep --dataset $DATASET --model_type madry --model_name adv_trained --attacker_type pgd --name $NAME/mnist/pgd --num_runs $NUM_RUNS --slurm $SLURM
#python -m online_attacks.scripts.online_attacks_sweep --dataset $DATASET --model_type madry --model_name adv_trained --attacker_type cw --name $NAME/mnist/cw --num_runs $NUM_RUNS --slurm $SLURM

# MNIST
DATASET="cifar"
BATCH_SIZE="256"
python -m online_attacks.scripts.online_attacks_sweep --dataset $DATASET --model_type madry --model_name adv_trained --attacker_type none --name $NAME/cifar/none --num_runs $NUM_RUNS --slurm $SLURM --batch_size $BATCH_SIZE
python -m online_attacks.scripts.online_attacks_sweep --dataset $DATASET --model_type madry --model_name adv_trained --attacker_type fgsm --name $NAME/cifar/fgsm --num_runs $NUM_RUNS --slurm $SLURM --batch_size $BATCH_SIZE
python -m online_attacks.scripts.online_attacks_sweep --dataset $DATASET --model_type madry --model_name adv_trained --attacker_type pgd --name $NAME/cifar/pgd --num_runs $NUM_RUNS --slurm $SLURM --batch_size $BATCH_SIZE
#python -m online_attacks.scripts.online_attacks_sweep --dataset $DATASET --model_type madry --model_name adv_trained --attacker_type cw --name $NAME/cifar/cw --num_runs $NUM_RUNS --slurm $SLURM --batch_size $BATCH_SIZE





