#!/usr/bin/env python3

from numpy.core.fromnumeric import amax, argmax
import torch
from torchvision.transforms.transforms import Resize
import wandb
import argparse

from torch.utils.data import DataLoader, ConcatDataset, Subset
from torchvision import datasets, transforms

from train import train
from models import BaselineCNN, BaselineCNN_w_dropout, NumDetector
from plotimage import plotimages
from dataloader import SVHNBackgroundImages, FullSVHN
from transforms import ResizeToFill

def main():
    model_options = {
        "BaselineCNN": BaselineCNN,
        "BaselineCNN_w_dropout": BaselineCNN_w_dropout,
        "NumDetector": NumDetector
    }

    optimizer_options = {
        "SGD": torch.optim.SGD,
        "Adam": torch.optim.Adam
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="What kind of model to use", type=str, choices=model_options.keys(), default="NumDetector")
    parser.add_argument("--optimizer", help="What kind of optimizer to use", type=str, choices=optimizer_options.keys(), default="SGD")
    parser.add_argument("--lr", help="Learning rate", type=float, default=1e-1)
    parser.add_argument("--epochs", help="Number of epochs", type=int, default=15)
    parser.add_argument("--batch-size", help="Batch size", type=int, default=64)
    parser.add_argument("--augmentation", help="Augmentation on or off", type=int, default=0)
    parser.add_argument("--workers", help="Number of workers for dataloading", type=int, default=8)
    parser.add_argument("--load", help="Path of trained model", type=str, default=None)
    parser.add_argument("--mode", help="Exlore or train", type=str, default="train")


    args = parser.parse_args()

    lr = args.lr
    epochs = args.epochs
    batch_size = args.batch_size
    augmentation = args.augmentation == 1

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
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.1)
        ]

    train_transform = transforms.Compose(train_transforms)
                             
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4381, 0.4442, 0.4732), (0.1170, 0.1200, 0.1025))
    ])


    train_background_data = SVHNBackgroundImages(
        train=True,
        transform=train_transform
    )

    trainset = ConcatDataset([
        datasets.SVHN(
            root="SVHN/train",
            split="train",
            transform=train_transform,
            download=True
        ),
        Subset(train_background_data, indices = range(0, len(train_background_data), 8))
    ])

    test_background_data = SVHNBackgroundImages(
        train=False,
        transform=test_transform
    )
    
    testset = ConcatDataset([
        datasets.SVHN(
            root="SVHN/test",
            split="test",
            transform=test_transform,
            download=True
        ),
        Subset(test_background_data, indices=range(0, len(test_background_data), 8))
    ])
    

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

    if args.load:
        model.load_state_dict(torch.load(args.load))

    model.to(device)

    wandb.watch(model)
    ## Weight decay?

    #Initialize the optimizer
    optimizer = optimizer_options[args.optimizer](model.parameters(), lr=lr)


    if args.mode == "train":
        return train(
            model=model,
            optimizer=optimizer,
            trainset=trainset,
            testset=testset,
            num_epochs=epochs,
            batch_size=batch_size,
            config=config,
            num_workers=args.workers
        )
    else:
        # Explore
        testset = FullSVHN(
            train = False,
            transform=transforms.Compose([
                ResizeToFill(width=128, height=64),
                test_transform
            ])
        )

        test_loader = DataLoader(testset, batch_size=1)

        for images, bboxes in test_loader:
            images = images.to(device)
            print(images.shape)
            output = model(images)
            
            argmax_output = output.argmax(dim=1)
            print(argmax_output)
            print()
            amax_output = output.amax(axis=1)
            print(amax_output.shape)
            print(amax_output*100)
            break


        pass
    

if __name__ == "__main__":
    main()