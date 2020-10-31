from advertorch.attacks import Attack
import torch
from tqdm import tqdm
from torch import optim
import ipdb
import sys
from __init__ import data_and_model_setup, load_data
from __init__ import eval as eval_attacker
import wandb
import torch
from torch import autograd
from torch.distributions.bernoulli import Bernoulli
import os
from advertorch.attacks import LinfPGDAttack
import argparse
from advertorch.context import ctx_noparamgrad_and_eval
from torchvision import transforms


sys.path.append("..")  # Adds higher directory to python modules path.
from utils import config as cf
from cnn_models import LeNet as Net
from attack_helpers import Linf_dist, L2_dist, ce_loss_func, save_image_to_wandb
from attack_helpers import attack_ce_loss_func, carlini_wagner_loss, non_saturating_loss, targeted_cw_loss
from models import create_generator
from utils.extragradient import Extragradient
import eval
from eval import baseline_transfer
from utils.utils import create_loaders, load_unk_model, nobox_wandb, CIFAR_NORMALIZATION
from classifiers import load_all_classifiers
from cnn_models.mnist_ensemble_adv_train_models import *
from defenses.ensemble_adver_train_mnist import *
kwargs_perturb_loss = {'Linf': Linf_dist, 'L2': L2_dist}
kwargs_attack_loss = {'cross_entropy': attack_ce_loss_func, 'carlini': carlini_wagner_loss,
                      'non_saturating': non_saturating_loss, 'targeted_cw': targeted_cw_loss}

def main():

    parser = argparse.ArgumentParser(description='NoBox')
    # Hparams
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate for the generator (default: 0.01)')
    parser.add_argument('--extragradient', default=False, action='store_true',
                        help='Use extragadient algorithm')
    parser.add_argument('--flow_layer_type', type=str, default='Linear',
                        help='Which type of layer to use ---i.e. GRevNet or Linear')
    parser.add_argument('--eval_set', default="test",
                        help="Evaluate model on test or validation set.")
    # Training
    parser.add_argument('--train_set', default='train',
                        choices=['train_and_test','test','train'],
                        help='add the test set in the training set')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--n_iter', type=int, default=500,
                        help='N iters for quere based attacks')
    parser.add_argument('--seed', type=int, metavar='S',
                        help='random seed (default: None)')
    parser.add_argument('--input_size', type=int, default=784, metavar='S',
                        help='Input size for MNIST is default')
    parser.add_argument('--batch_size', type=int, default=256, metavar='S',
                        help='Batch size')
    parser.add_argument('--attack_type', type=str, default='nobox',
                        help='Which attack to run')
    parser.add_argument('--attack_loss', type=str, default='cross_entropy',
                        help='Which loss func. to use to optimize G')
    parser.add_argument('--perturb_loss', type=str, default='L2', choices= ['L2','Linf'],
                        help='Which loss func. to use to optimize to compute constraint')
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--resample_iterations', type=int, default=100, metavar='N',
                        help='How many times to resample (default: 100)')
    parser.add_argument('--architecture', default="VGG16",
                        help="The architecture we want to attack on CIFAR.")
    parser.add_argument('--eval_freq', default=5, type=int,
                        help="Evaluate and save model every eval_freq epochs.")
    # Bells
    parser.add_argument("--wandb", action="store_true", default=False, help='Use wandb for logging')
    parser.add_argument('--namestr', type=str, default='NoBox', \
            help='additional info in output filename to describe experiments')
    parser.add_argument('--dir_test_models', type=str, default="./dir_test_models",
                        help="The path to the directory containing the classifier models for evaluation.")
    parser.add_argument('--normalize', default=None, choices=(None, "default", "meanstd"))

    ###
    parser.add_argument('--source_arch', default="res18",
                        help="The architecture we want to attack on CIFAR.")
    parser.add_argument('--target_arch', nargs='*',
                        help="The architecture we want to blackbox transfer to on CIFAR.")
    parser.add_argument('--adv_models', nargs='*', help='path to adv model(s)')
    parser.add_argument('--model_name', help='path to model')
    parser.add_argument('--transfer', action='store_true')
    parser.add_argument('--command', choices=("eval", "train"), default="train")
    parser.add_argument('--split', type=int, default=None,
                        help="Which subsplit to use.")
    parser.add_argument('--path_to_data', default="../data", type=str)
    args = parser.parse_args()

    args.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    normalize = None
    if args.normalize == "meanstd":
        normalize = transforms.Normalize(cf.mean["cifar10"], cf.std["cifar10"])
    elif args.normalize == "default":
        normalize = CIFAR_NORMALIZATION

    train_loader, test_loader, split_train_loader, split_test_loader = create_loaders(args,
            root=args.path_to_data, split=args.split, num_test_samples=args.num_test_samples, normalize=normalize)

    if args.split is not None:
        train_loader = split_train_loader
        test_loader = split_test_loader

    if os.path.isfile("../settings.json"):
        with open('../settings.json') as f:
            data = json.load(f)
        args.wandb_apikey = data.get("wandbapikey")

    if args.wandb:
        os.environ['WANDB_API_KEY'] = args.wandb_apikey
        wandb.init(project='Online-Attacks',
                   name='Online-Attack-{}-{}'.format(args.dataset, args.namestr))

    model, adv_models, l_test_classif_paths, args.model_type = data_and_model_setup(args, no_box_attack=True)
    model.to(args.dev)
    model.eval()

    print("Testing on %d Test Classifiers with Source Model %s" %(len(l_test_classif_paths), args.source_arch))
    x_test, y_test = load_data(args, test_loader)

    if args.dataset == "mnist":
        critic = load_unk_model(args)
    elif args.dataset == "cifar":
        name = args.source_arch
        if args.source_arch == "adv":
            name = "res18"
        critic = load_unk_model(args, name=name)

    misclassify_loss_func = kwargs_attack_loss[args.attack_loss]

    print("Evaluating clean error rate:")
    list_model = [args.source_arch]
    if args.source_arch == "adv":
        list_model = [args.model_type]
    if args.target_arch is not None:
        list_model = args.target_arch

    for model_type in list_model:
        num_samples = args.num_eval_samples
        if num_samples is None:
            num_samples = len(test_loader.dataset)

        eval_loader = torch.utils.data.Subset(test_loader.dataset,
                         np.random.randint(len(test_loader.dataset), size=(num_samples,)))
        eval_loader = torch.utils.data.DataLoader(eval_loader, batch_size=args.test_batch_size)
        baseline_transfer(args, None, "Clean", model_type, eval_loader,
            list_classifiers=l_test_classif_paths)

    def eval_fn(model):
        advcorrect = 0
        model.to(args.dev)
        model.eval()
        with ctx_noparamgrad_and_eval(model):
            if args.source_arch == 'googlenet':
                adv_complete_list = []
                for batch_idx, (x_batch, y_batch) in enumerate(test_loader):
                    if (batch_idx + 1) * args.test_batch_size > args.batch_size:
                        break
                    x_batch, y_batch = x_batch.to(args.dev), y_batch.to(args.dev)
                    adv_complete_list.append(attacker.perturb(x_batch, target=y_batch))
                adv_complete = torch.cat(adv_complete_list)
            else:
                adv_complete = attacker.perturb(x_test[:args.batch_size],
                                        target=y_test[:args.batch_size])
            adv_complete = torch.clamp(adv_complete, min=0., max=1.0)
            output = model(adv_complete)
            pred = output.max(1, keepdim=True)[1]
            advcorrect += pred.eq(y_test[:args.batch_size].view_as(pred)).sum().item()
            fool_rate = 1 - advcorrect / float(args.batch_size)
            print('Test set base model fool rate: %f' %(fool_rate))
        model.cpu()

        if args.transfer:
            adv_img_list = []
            y_orig = y_test[:args.batch_size]
            for i in range(0, len(adv_complete)):
                adv_img_list.append([adv_complete[i].unsqueeze(0), y_orig[i]])
            # Free memory
            del model
            torch.cuda.empty_cache()
            baseline_transfer(args, attacker, "AEG", model_type,
                            adv_img_list, l_test_classif_paths, adv_models)


    if args.command == "eval":
        attacker.load(args)
    elif args.command == "train":
        attacker.train(train_loader, test_loader, adv_models,
                l_test_classif_paths, l_train_classif={"source_model": model},
                eval_fn=eval_fn)

if __name__ == '__main__':
    main()
