import os
import torch

from dataloader import get_svhn
import wandb
from train import train
import argparse
import zipfile
import gdown
import torchvision.transforms as transforms

from models import BaselineCNN, BaselineCNN_w_dropout
from dataloader import get_svhn
from torchvision.datasets import SVHN

def main():

    model_options = {
        "BaselineCNN": BaselineCNN,
        "BaselineCNN_w_dropout": BaselineCNN_w_dropout,
        # "ResNet": ResNet,
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
    parser.add_argument("--augmentation", help="Augmentation on or off", type=bool, default=False)
    parser.add_argument("--workers", help="Number of workers for dataloading", type=bool, default=8)

    args = parser.parse_args()

    if torch.cuda.is_available():
        print("The code will run on GPU.")
    else:
        print("The code will run on CPU. Go to Edit->Notebook Settings and choose GPU as the hardware accelerator")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

###
    transform = []

    if args.augmentation == 1:
        transform.append(transforms.RandomRotation(20))
        transform.append(transforms.RandomHorizontalFlip())
        transform.append(transforms.ColorJitter(0.1, 0.1, 0.1, 0.1))
        print(args.augmentation)


    transform.append(transforms.ToTensor())
    transform.append(transforms.Normalize((0.4381, 0.4442, 0.4732), (0.1170, 0.1200, 0.1025)))
                

    train_transform = transforms.Compose(transform)
                                        
    test_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.4381, 0.4442, 0.4732), (0.1170, 0.1200, 0.1025))
                                        ])


###



    batch_size = 64
    trainset = get_svhn(train=True, transform=train_transform)
    testset = get_svhn(train=False, transform=test_transform)

    if not os.path.exists(f'./SVHN'):
            url = 'https://drive.google.com/file/d/1RWNq8JP5SXi07NLc1bgqN39FrsmB3KA7/view?usp=sharing'
            gdown.download(url, './SVHN.zip', quiet=False)
            try:
                with zipfile.ZipFile('./SVHN.zip') as z:
                    z.extractall("SVHN")
                    print("Extracted", 'SVHN.zip')
            except:
                print("Invalid file")

    lr = args.lr
    epochs = args.epochs

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

    train(
        model=model,
        optimizer=optimizer,
        trainset=trainset,
        testset=testset,
        num_epochs=epochs,
        batch_size=batch_size,
        save_weights=False,
        config=config,
        num_workers=args.workers
    )




if __name__ == "__main__":
    main()