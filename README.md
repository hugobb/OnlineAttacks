# Online Adversarial Attacks

Creating the python virtual environment `conda env create -f environment.yml`. 
After this you can type `source activate NoBox` to launch the environment.

## Folder Structure
```
attacks (folder with the attacks)
 ├── aeg.py
 ├── ...
 └── mi.py 
secretary_alg (folder with the different secretary algorithms)
 ├── optimistic.py
 ├── ...
 └── single.py
data_loading (loader for the online stream of data and the models)
 ├── load_unknown_model.py
 └── data_stream.py
utils (folder with the utils)
main.py 
```
