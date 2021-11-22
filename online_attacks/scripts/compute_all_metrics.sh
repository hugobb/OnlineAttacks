#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

NAME="/checkpoint/hberard/OnlineAttack/results_icml/madry-icml/"
SLURM="configs/learnfair.yaml"
NUM_RUNS="1000"

python -m online_attacks.scripts.eval_all $NAME/mnist/fgsm --dataset mnist --model_type madry --model_name secret --slurm $SLURM
python -m online_attacks.scripts.eval_all $NAME/mnist/pgd --dataset mnist --model_type madry --model_name secret --slurm $SLURM

python -m online_attacks.scripts.eval_all $NAME/cifar/fgsm  --dataset cifar --model_type madry --model_name secret --slurm $SLURM
python -m online_attacks.scripts.eval_all $NAME/cifar/pgd  --dataset cifar --model_type madry --model_name secret --slurm $SLURM
