import os
import gdown
import wandb
import zipfile
import argparse

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt


from utils import WeightedFocalLoss
from train import train
from models import BaselineUNet
from dataloader import LIDC_crops
from transforms import MultiHorizontalFlip, MultiRandomRotation, MultiRandomCrop, MultiToTensor, MultiNormalize


from utils import generalized_energy_distance

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
    parser.add_argument("--loss", help="Choose loss", type=str, choices=loss_options.keys(), default="BCE")
    parser.add_argument("--id", help="WandB ID tag to locate ensemble batch", type=str, default='basicRun')
    parser.add_argument("--paths", help="list of paths to ensemble models separated by comma", type=str, nargs="+", default=None)

    args = parser.parse_args()

    ensemble_id = args.id

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

    if args.paths:
        print('sampling image')

        # Model = model_options[args.model]
        
        # test_transform = transforms.Compose([MultiToTensor()])
        # image_transform = transforms.Compose([
        #     transforms.Normalize((0.4058,), (0.1222,))
        # ])

        # label_version = [0, 1, 2, 3]

        # test_set = LIDC_crops(test_transform, image_transform, mode = 'test', label_version = label_version)

        # test_loader = DataLoader(test_set, batch_size = batch_size, shuffle = True, num_workers = 8)

        # models = []
        # paths = args.paths[0].split(',')
        # for path in paths:
        #     model = Model(1, 128, 128)
        #     model.load_state_dict(torch.load(path, map_location=device))
        #     models.append(model)

        # ged = generalized_energy_distance(models, test_loader)
        # print(f"Generalized Energy Distance for ensamble model: {ged}")

        sample_image(model_options[args.model], args.paths[0].split(','), args.id)
    else:    
        models = ensemble(ensemble_id, 4, lr, batch_size, epochs, optimizer_options[args.optimizer], loss_options[args.loss], model_options[args.model], augmentation=args.augmentation)
        weigth_paths = ",".join([*models.keys()])
        print(f'String path for ensemble models:\n{weigth_paths}\n')




def ensemble(ensemble_id, label_versions, lr, batch_size, epochs, optimizer, loss_func, Net, num_workers=8, augmentation=0):
    models = {}
    for label_version in range(label_versions):
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

        train_set = LIDC_crops(base_transform, image_transform, label_version = [label_version])
        validation_set = LIDC_crops(test_transform, image_transform, mode = 'val', label_version = [label_version])
        test_set = LIDC_crops(test_transform, image_transform, mode = 'test', label_version = [label_version])

        train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 8)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #WANDB
        #1. Start a new run
        run = wandb.init(reinit=True, notes=f'r{label_version}', tags=[ensemble_id], 
                            project='lungs', entity='dlincv')

        # 2. Save model inputs and hyperparameters
        config = wandb.config
        config.learning_rate = lr
        config.batch_size = batch_size
        config.epochs = epochs
        config.optimizer = optimizer
        config.loss_func = loss_func
        config.transforms = image_transform


        input_shape = next(iter(train_loader))[0][0].shape
        # Init network
        model = Net(*input_shape)
        
        model.to(device)

        wandb.watch(model)

        #Initialize the optimizer
        optim = optimizer(model.parameters(), lr=lr)


        train(
            model=model,
            optimizer=optim,
            train_set=train_set,
            validation_set=validation_set,
            test_set=test_set,
            num_epochs=epochs,
            batch_size=batch_size,
            config=config,
            num_workers=num_workers,
            loss_func=loss_func
        )
        weights_path = os.path.join(wandb.run.dir,wandb.run.name + ".pth")
        models[weights_path] = model
        run.finish()
        print(f'Iteration {label_version} done')
    return models


def sample_image(Net, model_paths, ensemble_id):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_transform = MultiToTensor()
    image_transform = transforms.Normalize((0.4058,), (0.1222,))

    test_set = LIDC_crops(test_transform, image_transform, mode='test', label_version=[0])
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True, num_workers = 8)

    wandb.init(notes=f'', tags=[ensemble_id, 'ensemble_sample'], project='lungs', entity='dlincv')
    for ii in range(4):
        print(f'iter {ii}')
        image, _ = next(iter(test_loader))
        input_shape = image[0].shape

        model = Net(*input_shape)
        pred = torch.zeros(len(model_paths), image.shape[2], image.shape[3])
        for i, path in enumerate(model_paths):
            model.load_state_dict(torch.load(path))

            # model.to(device)
            model.eval()
            pred[i,:,:] = model(image)[0][0]


        fig, ax = plt.subplots(2,3, figsize=(9, 6))
        
        ax[0,0].imshow(image[0].numpy()[0], cmap="gray")
        ax[0,0].set_title('Input image')
        ax[0,0].axis("off")
        k = 0
        for i in range(2):
            for j in range(2):
                ax[j,i+1].imshow(pred[k].detach().cpu().numpy(), cmap="gray")
                ax[j,i+1].set_title(f"Model {k+1}")
                ax[j,i+1].axis("off")
                k += 1

        ax[1,0].imshow(pred.mean(dim=0).detach().cpu().numpy())
        ax[1,0].set_title(f"Mean of all predictions")
        ax[1,0].axis("off")

        fig.tight_layout()
        path = os.path.join('figs/', ensemble_id + str(ii) + '.png')
        fig.savefig(path)
        

        wandb.save(path)





if __name__ == "__main__":
    main()


