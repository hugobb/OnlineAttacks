
def run_experiment(args, K, train_loader):
    offline_algorithm, online_algorithm = create_online_algorithm(args, args.online_type, args.N, K)
    attacker = create_attacker(attacker_type, args)


    return comp_ratio

def main():
    parser = argparse.ArgumentParser(description='Online Attacks')
    # Hparams
    parser.add_argument('--K', type=int, default=1, metavar='K',
                        help='Number of attacks to submit')
    parser.add_argument('--N', type=int, default=5, metavar='N',
                        help='Size of datastream')
    parser.add_argument('--max_perms', type=int, default=120, metavar='P',
                        help='Maximum number of perms of the data stream')
    parser.add_argument('--seed', type=int, metavar='S',
                        help='random seed (default: None)')
    # Training
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--online_type', type=str, default='stochastic_virtual')
    # Bells
    parser.add_argument("--wandb", action="store_true", default=False, help='Use wandb for logging')
    parser.add_argument('--namestr', type=str, default='NoBox', \
            help='additional info in output filename to describe experiments')

    args = parser.parse_args()
    args.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_everything(args.seed)

    if os.path.isfile("settings.json"):
        with open('settings.json') as f:
            data = json.load(f)
        args.wandb_apikey = data.get("wandbapikey")

    if args.wandb:
        os.environ['WANDB_API_KEY'] = args.wandb_apikey
        wandb.init(project='Online-Attacks',
                   name='Online-Attack-{}-{}'.format(args.dataset, args.namestr))

    for k in range(1, args.K+1):
        train_loader = ToyDatastream(args.N, args.max_perms)
        comp_ratio = run_experiment(args, k, train_loader)
        if args.wandb:
            model_name = "Competitive Ratio " + args.online_type
            wandb.log({model_name: comp_ratio, "K": k})

if __name__ == '__main__':
    main()

