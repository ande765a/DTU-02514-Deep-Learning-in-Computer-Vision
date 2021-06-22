from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import torch
from torch.utils import data
import torch.nn as nn
from torchvision.utils import save_image, make_grid
import wandb

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
    




def one_img(test_loader, G_A2B, G_B2A, epochs, figpath = 'figs/'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for i in range(epochs):
        img_path = f"{figpath}one_image_{i}.png"
        A, B = next(iter(test_loader))
        original = A.to(device)
        fake = G_A2B(original) #horse to zebra
        recovered = G_B2A(fake) # fake zebra to horse
        identity = G_A2B(original) #horse to horse

        image_grid = torch.cat((original, fake, recovered, identity), 2)
        save_image(image_grid, img_path, normalize = False)
        wandb.save(img_path)


def plot_histogram(dataloader, discriminator, generator, batch_size, figpath='figs/', animal="horse",):
    
    if animal =="horse":
        A, B = next(iter(dataloader)) ## Horse
    else:
        B, A = next(iter(dataloader)) ## Zebra


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    A_real = A.to(device)*2-1 #scale to (-1, 1) range
    B_real = B.to(device)*2-1 
    d = discriminator
    g = generator
    z = torch.randn(batch_size, 100).to(device)

    A_hat = g(B_real)
    A_hat_pred = d(A_hat.detach())
    A_pred = d(A_real)

    real_tensor = torch.ones((A_real.shape[0], *d.output_shape)).to(device)
    fake_tensor = torch.zeros((A_real.shape[0], *d.output_shape)).to(device)

    GANLoss = nn.BCELoss
    # Discriminator losses
    loss_a_real_discriminator = GANLoss(A_pred, real_tensor)

    # TODO: use buffer here?
    loss_a_fake_discriminator = GANLoss(A_hat_pred, fake_tensor)

    # Combining generator losses
    loss_d_a = loss_a_real_discriminator + loss_a_fake_discriminator 

    discriminator_final_layer = torch.sigmoid
    H1 = discriminator_final_layer(d(g(z))).cpu()
    H2 = discriminator_final_layer(d(A_real)).cpu()
    plot_min = min(H1.min(), H2.min()).item()
    plot_max = max(H1.max(), H2.max()).item()
    plt.cla()
    plt.hist(H1.squeeze(), label='fake', range=(plot_min, plot_max), alpha=0.5)
    plt.hist(H2.squeeze(), label='real', range=(plot_min, plot_max), alpha=0.5)
    plt.legend()
    plt.set_xlabel('Probability of being real')
    plt.set_title('Discriminator loss: %.2f' % loss_d_a.item())
    img_path = f"{figpath}histogram_from_{animal}.png"
    plt.savefig(img_path=img_path)
    return img_path