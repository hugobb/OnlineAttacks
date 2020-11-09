import sys
sys.path.append("..")  # Adds higher directory to python modules path.
from online_algorithms import create_online_algorithm, compute_competitive_ratio
from datastream import load_dataset
from attacks import create_attacker


def run_experiment(args, K, train_loader):
    offline_algorithm, online_algorithm = create_online_algorithm(args, args.online_type, args.N, K)
    
    dataset = load_dataset(args.dataset)
    classifier = load_classifier(args)

    attacker = create_attacker(args.attacker, classifier, args)
    adversarial_dataset = 
    for i, dataset in enumerate(train_loader):
        comp_ratio_list.append(compute_competitive_ratio(dataset, online_algorithm, offline_algorithm)
    comp_ratio = np.sum(comp_ratio_list) / (K*num_perms)


    return comp_ratio