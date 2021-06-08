import os
import torch
from torch.utils.data import DataLoader
from dataloader import Hotdog_NotHotdog
import wandb
from train import train
import argparse
import zipfile
import gdown

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from models import BaselineCNN, BaselineCNN_w_dropout, ResNet
from dataloader import Hotdog_NotHotdog



def main():

    model_options = {
        "BaselineCNN": BaselineCNN,
        "BaselineCNN_w_dropout": BaselineCNN_w_dropout,
        "ResNet": ResNet,
    }

    optimixer_options = {
        "SGD": torch.optim.SGD,
        "Adam": torch.optim.Adam
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="What kind of model to use", type=str, choices=model_options.keys(), default="BaselineCNN")
    parser.add_argument("--optimizer", help="What kind of optimizer to use", type=str, choices=optimixer_options.keys(), default="SGD")
    parser.add_argument("--lr", help="Learning rate", type=float, default=1e-3)

    args = parser.parse_args()

    if torch.cuda.is_available():
        print("The code will run on GPU.")
    else:
        print("The code will run on CPU. Go to Edit->Notebook Settings and choose GPU as the hardware accelerator")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(f'./hotdog_nothotdog'):
            url = 'https://drive.google.com/uc?id=1-P-235zJdwk7joSxJ0Tw5eWCSHvVCiW7'
            gdown.download(url, './hotdog_nothotdog.zip', quiet=False)
            try:
                with zipfile.ZipFile('./hotdog_nothotdog.zip') as z:
                    z.extractall("hotdog_nothotdog")
                    print("Extracted", 'hotdog_nothotdog.zip')
            except:
                print("Invalid file")
    return

    size = 128
    train_transform = transforms.Compose([transforms.Resize((size, size)), 
                                        transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.Resize((size, size)), 
                                        transforms.ToTensor()])

    batch_size = 64
    trainset = Hotdog_NotHotdog(train=True, transform=train_transform)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testset = Hotdog_NotHotdog(train=False, transform=test_transform)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    lr = args.lr
    epochs = 10

    # WANDB 1. Start a new run
    wandb.init(project='hotdog', entity='dlincv')

    # 2. Save model inputs and hyperparameters
    config = wandb.config
    config.learning_rate = lr
    config.batch_size = batch_size
    config.epochs = epochs

    # Init network
    model = model_options[args.model]()
    model.to(device)
    wandb.watch(model)

    # Loss function
    criterion = nn.BCELoss()

    #Initialize the optimizer
    optimizer = optimizer_options[args.optimizer](lr=lr)



    train(model, optimizer, criterion, num_epochs, save_weights=False)


if __name__ == "__main__":
    main()