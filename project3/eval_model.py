import os
import wandb
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
from utils import plot_histogram, one_img
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def main():

    model_options = {
        "cycleGAN": {"Generator": Generator, "Discriminator": Discriminator}
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="What kind of model to use", type=str, choices=model_options.keys(), default="cycleGAN")
    parser.add_argument("--batch-size", help="Batch size", type=int, default=4)
    parser.add_argument("--augmentation", help="Augmentation on or off", type=int, default=0)
    parser.add_argument("--resize", help="Resize images to 128", type=int, default=0)
    parser.add_argument("--workers", help="Number of workers for dataloading", type=int, default=8)
    parser.add_argument("--load", help="Path of trained model", type=str, default=None)

    args = parser.parse_args()

    epochs = args.epochs
    batch_size = args.batch_size

    if torch.cuda.is_available():
        print("The code will run on GPU.")
    else:
        print("The code will run on CPU.")

    base_transform = []
    test_transform = []
    if args.resize == 1:
        size = 128
        base_transform.append(transforms.Resize((size, size)))
        test_transform.append(transforms.Resize((size, size)))
    
    if args.augmentation == 1:
        base_transform.append(transforms.RandomRotation(5))
        base_transform.append(transforms.RandomHorizontalFlip())
        base_transform.append(transforms.ColorJitter(0.1, 0.1, 0.1, 0.1))
    
    base_transform.append(transforms.ToTensor())
    test_transform.append(transforms.ToTensor())
    
    base_transform = transforms.Compose(base_transform)
    test_transform = transforms.Compose(test_transform)

    test_set = horse2zebra(test_transform, train=False)

    model_G = model_options[args.model]["Generator"]
    model_D = model_options[args.model]["Discriminator"]

    wandb.init(project='horse2zebra', entity='dlincv')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    config = wandb.config
    config.batch_size 

    img_shape = next(iter(test_loader))[0][0].shape

    # Setup Generator for A to B and discriminator for B
    G_a2b = model_G(blocks=9).to(device)
    D_b = model_D(tuple(img_shape)).to(device)
    wandb.watch(G_a2b)
    wandb.watch(D_b)

    # Setup Generator for B to A and discriminator for A
    G_b2a = model_G(blocks=9).to(device)
    D_a = model_D(tuple(img_shape)).to(device)
    wandb.watch(D_a)
    wandb.watch(G_b2a)

    G_a2b.eval()
    G_a2b.eval()
    D_a.eval()
    D_b.eval()

    test_loss_GAN_total_epoch = []
    test_loss_D_epoch = []
    # Test loop
    print(f"len test set {len(test_set)}")
    img_path = plot_histogram(test_loader, D_b, G_a2b, batch_size)
    wandb.save(img_path)
    img_path = plot_histogram(test_loader, D_a, G_b2a, batch_size, animal="fuckingZebra2fuckingHorze")
    wandb.save(img_path)
    one_img(test_loader, G_a2b, G_b2a,epochs=len(test_set))
    

if __name__ == "__main__": 
    main()
    