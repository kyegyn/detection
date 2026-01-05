from utils import seed_torch

import torch
import argparse

def train(args):
    pass

def test(args):
    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Deep Fake Detection")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--phase", type=str, default="test", choices=["train", "test"], help="Phase of the experiment")
    parser.add_argument("--trainset_dirpath", type=str, default="data/train", help="Trainset directory")
    parser.add_argument("--testset_dirpath", type=str, default="data/test", help="Testset directory")
    parser.add_argument("--ckpt", type=str, default="ckpt", help="Checkpoint directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--seed", type=int, default=1024, help="Random seed")

    args = parser.parse_args()

    # Set the random seed
    seed_torch(args.seed)

    # Set the GPU ID
    args.device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    # Begin the experiment
    if args.phase == "train":
        train(args)
    elif args.phase == "test":
        test(args)
    else:
        raise ValueError("Unknown phase")
