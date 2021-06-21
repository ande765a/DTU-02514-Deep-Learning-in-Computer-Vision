import matplotlib.pyplot as plt
import torch
from torchvision.utils import save_image, make_grid

def plotimages(test_loader, model, figname, figpath = 'figs/'):
    horse_im, zebra_im = next(iter(test_loader))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    output = model(horse_im.to(device))
    fig, ax = plt.subplots(2, 5, figsize = (10,5))

    for i in range(5):
        ax[0,i].imshow(horse_im)
        ax[1,i].imshow(output)
    fig.savefig(figpath + figname)

    return figpath

def sample_images(test_loader, G_A2B, G_B2A, epoch, figpath = 'figs/'):
    """Saves a generated sample from the test set"""
    img_path = f"{figpath}images{epoch}.png"
    A, B = next(iter(test_loader))
    G_A2B.eval()
    G_B2A.eval()
    real_A = A
    fake_B = G_A2B(real_A)
    real_B = B
    fake_A = G_B2A(real_B)
    # Arrange x-axis
    real_A = make_grid(real_A, nrow=5, normalize=True)
    real_B = make_grid(real_B, nrow=5, normalize=True)
    fake_A = make_grid(fake_A, nrow=5, normalize=True)
    fake_B = make_grid(fake_B, nrow=5, normalize=True)
    # Arrange y-axis
    image_grid = torch.cat((torch.cat((real_A, fake_B), 1), torch.cat((real_B, fake_A),1)))
    save_image(image_grid, img_path, normalize=False)
    return img_path
    
