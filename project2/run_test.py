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
from models import BaselineUNet, BaselineUNetDropout
from dataloader import LIDC_crops
from transforms import MultiHorizontalFlip, MultiRandomRotation, MultiRandomCrop, MultiToTensor, MultiNormalize
from utils import plotimages, measures, run_test

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def main():

    model_options = {
        "BaselineUNet": BaselineUNet,
        "BaselineUnetDropout": BaselineUNetDropout
    }

    loss_options = {
        "BCE": nn.BCEWithLogitsLoss,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="What kind of model to use", type=str, choices=model_options.keys(), default="BaselineUNet")
    parser.add_argument("--batch-size", help="Batch size", type=int, default=64)
    parser.add_argument("--augmentation", help="Augmentation on or off", type=int, default=0)
    parser.add_argument("--workers", help="Number of workers for dataloading", type=int, default=8)
    parser.add_argument("--load", help="Path of trained model", type=str, default=None)
    parser.add_argument("--loss", help="Choose loss", type=str, choices=loss_options.keys(), default="BCE")
    parser.add_argument("--labels", help="Train on whole set of labels", type=int, nargs="+", default=[0,1,2,3])

    args = parser.parse_args()

    batch_size = args.batch_size
    augmentation = args.augmentation
    label_version = args.labels


    if torch.cuda.is_available():
        print("The code will run on GPU.")
    else:
        print("The code will run on CPU.")
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    base_transform = [MultiToTensor()]

    if augmentation == 1: 
        base_transform += [
            MultiRandomCrop((128, 128)),
            MultiHorizontalFlip(), 
            MultiRandomRotation(20)
        ]

    
    base_transform = transforms.Compose(base_transform)
    test_transform = transforms.Compose([MultiToTensor()])
    image_transform = transforms.Compose([
        transforms.Normalize((0.4058,), (0.1222,))
    ])

    test_set = LIDC_crops(test_transform, image_transform, mode = 'test', label_version = label_version)

    test_loader = DataLoader(test_set, batch_size = batch_size, shuffle = True, num_workers = 8)

    #WANDB
    #1. Start a new run
    wandb.init(tags=['test_data_run'], project='lungs', entity='dlincv')

    # 2. Save model inputs and hyperparameters
    config = wandb.config
    config.batch_size = batch_size
    config.loss_func = args.loss
    config.transforms = image_transform

    input_shape = next(iter(test_loader))[0][0].shape
    # Init network
    model = model_options[args.model](*input_shape)
    

    wandb.watch(model)

    if args.load:
        model.to(device)
        model.load_state_dict(torch.load(args.load))
        criterion = loss_options[args.loss]
        run_test(model, test_loader=test_loader, criterion=criterion(), config=config)





if __name__ == "__main__":
    main()
