import os
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.transforms import Normalize
from dataloader import Hotdog_NotHotdog
import wandb
from train import train
import argparse
import zipfile
import gdown
# from medcam import medcam


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

    size = 128

    transform = [transforms.Resize((size, size))]

    if args.augmentation:
        transform.append(transforms.RandomRotation(20))
        transform.append(transforms.RandomHorizontalFlip())
        transform.append(transforms.ColorJitter(0.1, 0.1, 0.1, 0.1))


    transform.append(transforms.ToTensor())
    transform.append(transforms.Normalize(mean=(0.5226, 0.4412, 0.3585), std=(0.2253, 0.2294, 0.2339)))
                

    train_transform = transforms.Compose(transform)
                                        
    test_transform = transforms.Compose([transforms.Resize((size, size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.5226, 0.4412, 0.3585),
                                                            std=(0.2253, 0.2294, 0.2339))
                                        ])

    batch_size = 64
    trainset = Hotdog_NotHotdog(train=True, transform=train_transform)
    testset = Hotdog_NotHotdog(train=False, transform=test_transform)

    lr = args.lr
    epochs = args.epochs

    # WANDB 1. Start a new run
    wandb.init(project='hotdog', entity='dlincv')

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
        config=config
    )




def analyze_data(trainset, batch_size):
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)

    mean = 0.
    std = 0.
    for images, _ in train_loader:
        batch_samples = images.size(0) # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= len(train_loader.dataset)
    std /= len(train_loader.dataset)  
    
    print(mean, std)

    # 
    
    # return filename
    

if __name__ == "__main__":
    main()