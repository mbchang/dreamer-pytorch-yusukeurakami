import argparse

def build_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--nbatches', type=int, default=int(1e6))
    parser.add_argument('--num_slots', type=int, default=5)
    parser.add_argument('--iters', type=int, default=3)
    parser.add_argument('--visualize_every', type=int, default=5000)
    parser.add_argument('--slot_temp', type=float, default=1)
    parser.add_argument('--obs_temp', type=float, default=10)
    parser.add_argument('--kl_coeff', type=float, default=1e-4)
    parser.add_argument('--slot_dim', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--subroot', type=str, default='figs')
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--printf', action='store_true')
    return parser

def create_expname(args):
    expname = ''
    expname += '_otemp{}'.format(args.obs_temp)
    expname += '_bsize{}'.format(args.batch_size)
    expname += '_klcoeff{}'.format(args.kl_coeff)
    expname += '_slotdim{}'.format(args.slot_dim)
    expname += '_lr{}'.format(args.lr)
    return expname