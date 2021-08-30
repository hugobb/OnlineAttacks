# Online Adversarial Attacks

Creating the python virtual environment `conda env create -f environment.yml`. 
After this you can type `source activate NoBox` to launch the environment.

## Requirements
Requuires to install:
- `omegaconf`
- `nestargs`

It's also useful to add the `online_adversarial_attacks` path to your `PYTHONPATH`.
Some scripts might not work otherwise.


## Command to run Toy experiment
The import might not work, and maybe need to add current directory to `PYTHONPATH`
`python online_attacks/experiments/toy.py --online_params.online_type stochastic_optimistic --online_params.N 100 --max_perms 1000 --K 10 `

## Command to run Mnist experiment (work in progress)
Currently load a random classifier and only evaluates competitive ratio and not attack success rate.
`python online_attacks/experiments/mnist_online.py --params.online_params.online_type stochastic_optimistic --params.online_params.K 100`

**TODO**:
- Right now the different algorithms are evaluated on different permutations, should we evaluate them on the same permuations ? (We can use a fixed random generator to generate the permutations, this should enforce that the permuation are the same) or (we can generate a list of permutations in a txt file, and load it) -> I prefer the 2nd solution.  

## Folder Structure
```
online_attacks
 ├── attacks (folder with the attacks)
      ├── mi.py
      ├── ...
      └── pgd.py
 ├── classifiers (folder with the different classifiers to load)
      ├── mnist
            ├── dataset.py
            ├── model.py
            ├── params.py
            └── train.py
      ├── ...
      └── launch.py (script to train the different classifiers)
 ├── data_loading (loader for the online stream of data)
      ├── toy_data.py
      └── data_stream.py
 ├── experiments (folder with the scripts to launch the different experiments)
      ├── mnist_online.py
      └── data_stream.py
 ├── online_algorithms (folder with the different secretary algorithms)
      ├── offline_algorithm.py
      ├── ...
      └── stochastic_virtual.py

 ├── utils (folder with the utils)
 └── launcher.py
```
