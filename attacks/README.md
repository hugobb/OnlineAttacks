# Online Attacks

This contains sample commands to run all of our Baseline models, taken from the
original git repos but repurposed for the Online Attacks codebase.

## Requirements
```
wandb (latest version) \
pytorch==1.4 \
torchvision \
cudatoolkit==10.1\
PIL \
numpy \
json \
wandb \
tqdm \
matplotlib \
image \
ipdb \
[advertorch](https://github.com/BorealisAI/advertorch) \
robustml
```

## Baseline split eval:
```
python momentum_iterative_attack.py --n_iter=1 --dataset cifar --epsilon=0.03125 --transfer --source_arch wide_resnet --split 0
```

## MI Attack
```
python momentum_iterative_attack.py --n_iter=100 --dataset cifar --epsilon=0.03125 --transfer
```

## DIM Attack
```
python diverse_input_attack.py --n_iter=100 --dataset cifar --epsilon=0.03125 --transfer
```

## SGM Attack
```
python sgm_attack.py --n_iter=20 --dataset cifar --transfer
```

## License
[MIT](https://choosealicense.com/licenses/mit/)


