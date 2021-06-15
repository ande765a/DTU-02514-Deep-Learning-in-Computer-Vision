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
from utils import WeightedFocalLoss
from train import train
from models import BaselineUNet
from dataloader import LIDC_crops

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def main():

    model_options = {
        "BaselineUNet": BaselineUNet
    }

    optimizer_options = {
        "Adam": torch.optim.Adam,
        "SGD": torch.optim.SGD
    }

    loss_options = {
        "BCE": nn.BCEWithLogitsLoss,
        "Focal": WeightedFocalLoss
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="What kind of model to use", type=str, choices=model_options.keys(), default="BaselineUNet")
    parser.add_argument("--optimizer", help="What kind of optimizer to use", type=str, choices=optimizer_options.keys(), default="Adam")
    parser.add_argument("--lr", help="Learning rate", type=float, default=1e-3)
    parser.add_argument("--epochs", help="Number of epochs", type=int, default=15)
    parser.add_argument("--batch-size", help="Batch size", type=int, default=128)
    parser.add_argument("--augmentation", help="Augmentation on or off", type=int, default=0)
    parser.add_argument("--workers", help="Number of workers for dataloading", type=int, default=8)
    parser.add_argument("--load", help="Path of trained model", type=str, default=None)
    parser.add_argument("--loss", help="Choose loss", type=str, choices=loss_options.keys(), default="BCE")

    args = parser.parse_args()

    lr = args.lr
    epochs = args.epochs
    batch_size = args.batch_size
    augmentation = args.augmentation == 1

    if torch.cuda.is_available():
        print("The code will run on GPU.")
    else:
        print("The code will run on CPU.")
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(f'./LIDC_crops'):
            url = 'https://drive.google.com/uc?id=1ggXR0MTfAk8Tq_tWu9kpCjHsuc6J6D3z'
            gdown.download(url, './LIDC_crops.zip', quiet=False)
            try:
                with zipfile.ZipFile('./LIDC_crops.zip') as z:
                    z.extractall("LIDC_crops")
                    print("Extracted", 'LIDC_crops.zip')
            except Exception as e:
                print(f"Invalid file {e}")

    img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4058,), (0.1222,))])
    label_transform = transforms.ToTensor()

    train_set = LIDC_crops(img_transform, label_transform)
    validation_set = LIDC_crops(img_transform, label_transform, mode = 'val')
    test_set = LIDC_crops(img_transform, label_transform, mode = 'test')


    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 8)

    #WANDB
    #1. Start a new run
    wandb.init(project='lungs', entity='dlincv')

    # 2. Save model inputs and hyperparameters
    config = wandb.config
    config.learning_rate = lr
    config.batch_size = batch_size
    config.epochs = epochs
    config.optimizer = optimizer_options[args.optimizer]
    config.loss_func = args.loss
    config.transforms = img_transform


    input_shape = next(iter(train_loader))[0][0].shape
    # Init network
    model = model_options[args.model](*input_shape)
    
    model.to(device)

    wandb.watch(model)

    #Initialize the optimizer
    optimizer = optimizer_options[args.optimizer](model.parameters(), lr=lr)


    if args.load:
        model.load_state_dict(torch.load(args.load))
    else:
        train(
            model=model,
            optimizer=optimizer,
            train_set=train_set,
            validation_set=validation_set,
            test_set=test_set,
            num_epochs=epochs,
            batch_size=batch_size,
            config=config,
            num_workers=args.workers,
            loss_func=loss_options[args.loss]
        )

    return




if __name__ == "__main__":
    main()
