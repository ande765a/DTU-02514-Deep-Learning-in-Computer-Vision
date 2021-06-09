import matplotlib.pyplot as plt
import torch
import numpy as np

def plotimages(dataloader):
    images, labels = next(iter(dataloader))
    plt.figure(figsize=(20,10))
    for i in range(21):
        plt.subplot(5,7,i+1)
        plt.imshow(images[i].numpy()[0], 'gray')
        plt.title(labels[i].item())
        plt.axis('off')
    plt.show()
    path = 'figs/svhnImg.png'
    plt.savefig(path)
    
    return path

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

def plotwrongimages(test_loader, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    plt.figure(figsize=(20,10))
    wrong_images = []
    for images, labels in test_loader:
        for image, label in zip(images, labels):
            if len(wrong_images) > 21:
                break
            unnorm = UnNormalize((0.4381, 0.4442, 0.4732), (0.1170, 0.1200, 0.1025))
            
            image = image.view(1, *image.shape)
            pred = model(image.to(device)).detach().cpu().numpy().item()
            label.item()

            if label.item() != (pred >0.5) and label.item() == 0:
                wrong_images.append((unnorm(image).permute(0,2,3,1).detach().cpu().numpy()[0], label, pred))

    print(len(wrong_images))
    for i in range(len(wrong_images)):
        plt.subplot(5,7,i+1)
        image, label, pred = wrong_images[i]
        plt.imshow(image)
        plt.title(f"Predicted: {pred:.2f}, True: {label}")
        plt.axis('off')
    plt.show()

    path = 'figs/wrongpred.png'
    
    plt.savefig(path)
    return path


