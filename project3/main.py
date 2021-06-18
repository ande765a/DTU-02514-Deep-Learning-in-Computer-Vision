import os
import gdown
import wandb
import zipfile
import argparse

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
# from utils import WeightedFocalLoss
from train import train
from models import Generator, Discriminator 
from dataloader import horse2zebra
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def main():

    model_options = {
        "cycleGAN": {"Generator": Generator, "Discriminator": Discriminator}
    }

    optimizer_options = {
        "Adam": torch.optim.Adam,
        "SGD": torch.optim.SGD
    }

    loss_options = {
        "LSGAN": {"GAN":nn.BCELoss, "Cycle": nn.L1Loss, "Identity": nn.L1Loss},
        # "Focal": WeightedFocalLoss
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="What kind of model to use", type=str, choices=model_options.keys(), default="cycleGAN")
    parser.add_argument("--optimizer", help="What kind of optimizer to use", type=str, choices=optimizer_options.keys(), default="Adam")
    parser.add_argument("--lr", help="Learning rate", type=float, default=1e-3)
    parser.add_argument("--epochs", help="Number of epochs", type=int, default=15)
    parser.add_argument("--batch-size", help="Batch size", type=int, default=128)
    #parser.add_argument("--augmentation", help="Augmentation on or off", type=int, default=0)
    parser.add_argument("--workers", help="Number of workers for dataloading", type=int, default=8)
    parser.add_argument("--load", help="Path of trained model", type=str, default=None)
    parser.add_argument("--loss", help="Choose loss", type=str, choices=loss_options.keys(), default="LSGAN")

    args = parser.parse_args()

    lr = args.lr
    epochs = args.epochs
    batch_size = args.batch_size
    augmentation = args.augmentation


    if torch.cuda.is_available():
        print("The code will run on GPU.")
    else:
        print("The code will run on CPU.")
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists(f'./horse2zebra'):
            url = 'https://drive.google.com/uc?id=1DThH7JGgAXrnMHqjFO5emgDdiNxvFCjA'
            
            gdown.download(url, './horse2zebra.zip', quiet=False)
            try:
                with zipfile.ZipFile('./horse2zebra.zip') as z:
                    z.extractall(".")
                    print("Extracted", 'horse2zebra.zip')
            except Exception as e:
                print(f"Invalid file {e}")


    size = 128
    base_transform = [transforms.Resize((size, size)), transforms.ToTensor()]
    test_transform = [transforms.Resize((size, size)), transforms.ToTensor()]
    # if augmentation == 1: 
    #     base_transform += [
    #         MultiRandomCrop((128, 128)),
    #         MultiHorizontalFlip(), 
    #         MultiRandomRotation(20)
    #     ]

    
    base_transform = transforms.Compose(base_transform)
    test_transform = transforms.Compose(test_transform)

    train_set = horse2zebra(base_transform)
    test_set = horse2zebra(test_transform)



    

if __name__ == "__main__": 
    main()
    