import os
import torch
import wandb
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataloader import LIDC_crops

from torch.utils.data import Subset

from transforms import MultiToTensor
from argparse import ArgumentParser
from models import BaselineUNetDropout
from utils import generalized_energy_distance
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def montecarlo(model, image, num_loop = 1000):
    pred = torch.zeros(num_loop, image.shape[0], image.shape[1])
    for i in num_loop:
        temp, _ = model(image) # Predict image, while dropout true
        pred[i,:,:] = temp[0]/num_loop #Weighted by 1/num_loop
    return torch.sum(pred, axis = 0)



def montecarlo_sample_image(model, ensemble_id, test_loader, num_iterations=1000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image, *label = next(iter(test_loader))
    input_shape = image[0].shape

    fig, ax = plt.subplots(1,2, figsize=(10, 6))
    
    ax[0].imshow(image[0].numpy()[0], cmap="gray")
    ax[0].set_title('Input image')
    ax[0].axis("off")
    pred = torch.zeros(num_iterations, image.shape[2], image.shape[3])
    image = image.to(device)
    model.train()
    for i in range(num_iterations):
        print(i)
        temp, _ = model(image) # Predict image, while dropout true
        pred[i,:,:] = temp[0] #Weighted by 1/num_loop

    ax[1].imshow(pred.mean(dim=0).detach().cpu().numpy())
    ax[1].set_title(f"Mean of {num_iterations} Monte Carlo predictions")
    ax[1].axis("off")

    # ax[2].imshow(label[0].numpy()[0][0])
    # ax[2].set_title(f"Ground truth from set 0")
    # ax[2].axis("off")


    fig.tight_layout()
    path = os.path.join('figs/', ensemble_id + '.png')
    fig.savefig(path)
    
    wandb.init(notes=f'', tags=[ensemble_id, 'ensemble_sample'], project='lungs', entity='dlincv')

    wandb.save(path)

def main():
    parser = ArgumentParser()
    parser.add_argument("--load", help="Path of trained model", type=str, default=None)
    parser.add_argument("--num-iterations", help="Number of times to resample from dropout model", type=int, default=200)

    args = parser.parse_args()
    model = BaselineUNetDropout(1, 128, 128)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)


    test_transform = MultiToTensor()
    image_transform = transforms.Normalize((0.4058,), (0.1222,))

    test_set = LIDC_crops(test_transform, image_transform, mode='test', label_version=[0])
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True, num_workers = 8)


    # Load model
    model.load_state_dict(torch.load(args.load, map_location=device))
    montecarlo_sample_image(model, ensemble_id="Ignore_me_juice", num_iterations=args.num_iterations, test_loader=test_loader)

    reduced_test_set = LIDC_crops(test_transform, image_transform, mode='test')
    reduced_test_set = Subset(reduced_test_set, indices=range(0, len(test_set) // 5))
    reduced_test_loader = DataLoader(reduced_test_set, batch_size=32, shuffle=True, num_workers = 8)

    ged = generalized_energy_distance([model], reduced_test_loader)
    
    print(f"GED: {ged}")

if __name__ == "__main__":
    main()