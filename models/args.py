import os
from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser(description="PyTorch deep learning models for document classification")
    par = '/home/projects/'
    parser.add_argument('--no-cuda', action='store_false', dest='cuda')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=3435)  # default value: 3435
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--log-every', type=int, default=10)
    parser.add_argument('--data-dir', default=os.path.join(par, 'hedwig-data', 'datasets')) #os.pardir
    parser.add_argument('--early_on_f1', action='store_true')

    return parser
