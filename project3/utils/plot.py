from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import torch
from torchvision.utils import save_image, make_grid

def sample_images(test_loader, G_A2B, G_B2A, epoch, figpath = 'figs/'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    """Saves a generated sample from the test set"""
    img_path = f"{figpath}images{epoch}.png"
    print(img_path)
    A, B = next(iter(test_loader))
    G_A2B.eval()
    G_B2A.eval()
    real_A = A.to(device)
    fake_B = G_A2B(real_A)
    real_B = B.to(device)
    fake_A = G_B2A(real_B)
    # Arrange x-axis
    real_A = make_grid(real_A, nrow=5, normalize=True)
    real_B = make_grid(real_B, nrow=5, normalize=True)
    fake_A = make_grid(fake_A, nrow=5, normalize=True)
    fake_B = make_grid(fake_B, nrow=5, normalize=True)
    # Arrange y-axis
    image_grid = torch.cat((torch.cat((real_A, fake_B), 1), torch.cat((real_B, fake_A),1)), 2)

    save_image(image_grid, img_path, normalize=False)
    return img_path
    

def one_img(test_loader, G_A2B, G_B2A, epoch, figpath = 'figs/'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_path = f"{figpath}one_image_{epoch}.png"
    A, B = next(iter(test_loader))
    original = A.to(device)
    fake = G_A2B(original) #horse to zebra
    recovered = G_B2A(fake) # fake zebra to horse
    identity = G_A2B(original) #horse to horse

    image_grid = torch.cat((original, fake, recovered, identity),2)
    save_image(image_grid, img_path, normalize = False)
    return