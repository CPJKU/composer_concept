""" Slightly adapted from original repo (https://github.com/KimSSung/Deep-Composer-Classification/blob/master/config.py) """

# Configuration file for (base training / adversarial training / adversarial attack)

import argparse
from distutils import util

# Basic Configuration
parser = argparse.ArgumentParser(description="Base Training")


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed


parser.add_argument(
    "--mode",
    default="basetrain",  # force to input mode ^_^
    type=str,
    required=True,
    choices=['basetrain']
    #    help="Mode (basetrain / advtrain / attack / generate / convert / split)",
)

parser.add_argument(
    "--model_name",
    type=str,
    default="resnet50",
    help="Prefix of model name (resnet18 / resnet34 / resnet50 / resnet101 / resnet152 / convnet)",
)

parser.add_argument(
    "--optim",
    type=str,
    default="SGD",
    help="Optimizer [Adadelta, Adagrad, Adam, AdamW, SparseAdam, Adamax, ASGD, RMSprop, Rprop, SGD, Nesterov]",
)

parser.add_argument(
    "--transform", type=str, default=None, help="Transform mode [Transpose / Tempo]",
)
parser.add_argument(
    "--epochs",
    default=100,
    type=int,
    help="Total number of epochs to run. Not actual epoch.",
)

parser.add_argument(
    "--save_trn", type=lambda x: bool(util.strtobool(x)), default=True, help="Whether to save both model & loader."
)

parser.add_argument("--lr", default=0.01, type=float, help="Model learning rate.")

## additional args used by trainer

parser.add_argument(
    "--onset", type=lambda x:bool(util.strtobool(x)), default=True, help="use Onset channel"
)

parser.add_argument(
    "--trn_seg", default=90, type=int, help="segment number for train."
)

parser.add_argument(
    "--val_seg", default=90, type=int, help="segment number for valid."
)

parser.add_argument(
    "--omit", default=None, type=str, help="List of omitted composers' indices.",
)

parser.add_argument(
    "--composers", default=13, type=int, help="The number of composers.",
)

parser.add_argument(
    "--age", default=False, type=lambda x:bool(util.strtobool(x)), help="Classification of Age? (True / False)",
)

parser.add_argument(
    "--train_batch", default=40, type=int, help="Batch size for training"
)