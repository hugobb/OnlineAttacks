#!/bin/bash

CURRENT_PATH=$(pwd)
MODELS_PATH=/checkpoint/hberard/OnlineAttack/pretained_models
MADRY_MNIST_PATH=online_attacks/classifiers/madry/madry_mnist
MADRY_CIFAR_PATH=online_attacks/classifiers/madry/madry_cifar

### INSTALLING MADRY MNIST CHALLENGE ###
git clone https://github.com/MadryLab/mnist_challenge $MADRY_MNIST_PATH
mkdir /tmp/madry_mnist
cd /tmp/madry_mnist
python $CURRENT_PATH/$MADRY_MNIST_PATH/fetch_model.py secret
python $CURRENT_PATH/$MADRY_MNIST_PATH/fetch_model.py natural
python $CURRENT_PATH/$MADRY_MNIST_PATH/fetch_model.py adv_trained
mkdir $MODELS_PATH/mnist/madry
mv models/* $MODELS_PATH/mnist/madry
cd $CURRENT_PATH
rm -rf /tmp/madry_mnist

### INSTALLING MADRY CIFAR CHALLENGE ###
git clone https://github.com/MadryLab/cifar10_challenge $MADRY_CIFAR_PATH
mkdir /tmp/madry_cifar
cd /tmp/madry_cifar
python $CURRENT_PATH/$MADRY_CIFAR_PATH/fetch_model.py secret
python $CURRENT_PATH/$MADRY_CIFAR_PATH/fetch_model.py natural
python $CURRENT_PATH/$MADRY_CIFAR_PATH/fetch_model.py adv_trained

mv models/model_0 models/secret
mv models/naturally_trained models/natural

mkdir $MODELS_PATH/cifar/madry
mv models/* $MODELS_PATH/cifar/madry
cd $CURRENT_PATH
rm -rf /tmp/madry_cifar