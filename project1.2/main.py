#!/usr/bin/env python3

import torch
import wandb
import argparse

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from train import train
from models import BaselineCNN, BaselineCNN_w_dropout
from plotimage import plotimages

def main():
    model_options = {
        "BaselineCNN": BaselineCNN,
        "BaselineCNN_w_dropout": BaselineCNN_w_dropout,
    }

    optimizer_options = {
        "SGD": torch.optim.SGD,
        "Adam": torch.optim.Adam
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="What kind of model to use", type=str, choices=model_options.keys(), default="BaselineCNN")
    parser.add_argument("--optimizer", help="What kind of optimizer to use", type=str, choices=optimizer_options.keys(), default="SGD")
    parser.add_argument("--lr", help="Learning rate", type=float, default=1e-3)
    parser.add_argument("--epochs", help="Number of epochs", type=int, default=10)
    parser.add_argument("--batch-size", help="Batch size", type=int, default=64)
    parser.add_argument("--augmentation", help="Augmentation on or off", type=bool, default=False)

    args = parser.parse_args()

    lr = args.lr
    epochs = args.epochs
    batch_size = args.batch_size
    augmentation = args.augmentation

    if torch.cuda.is_available():
        print("The code will run on GPU.")
    else:
        print("The code will run on CPU.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.4381, 0.4442, 0.4732), (0.1170, 0.1200, 0.1025)), 
    ]

    if augmentation:
        train_transforms += [
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.1)
        ]

    train_transform = transforms.Compose(train_transforms)
                             
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4381, 0.4442, 0.4732), (0.1170, 0.1200, 0.1025))
    ])


    trainset = datasets.SVHN(
        root="SVHN/train",
        split="train",
        transform=train_transform,
        download=True
    )
    
    testset = trainset = datasets.SVHN(
        root="SVHN/test",
        split="test",
        transform=test_transform,
        download=True
    )


    # WANDB 1. Start a new run
    wandb.init(project='numDetection', entity='dlincv')

    # 2. Save model inputs and hyperparameters
    config = wandb.config
    config.learning_rate = lr
    config.batch_size = batch_size
    config.epochs = epochs
    config.optimizer = optimizer_options[args.optimizer]
    config.transforms = train_transform
    
    # Init network
    model = model_options[args.model]()
    model.to(device)
    wandb.watch(model)
    ## Weight decay?

    #Initialize the optimizer
    optimizer = optimizer_options[args.optimizer](model.parameters(), lr=lr)


    return train(
        model=model,
        optimizer=optimizer,
        trainset=trainset,
        testset=testset,
        num_epochs=epochs,
        batch_size=batch_size,
        config=config
    )
    

if __name__ == "__main__":
    main()