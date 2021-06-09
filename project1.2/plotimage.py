import matplotlib.pyplot as plt
import torch


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


def plotwrongimages(test_loader, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    plt.figure(figsize=(20,10))

    wrong_images = []
    for images, labels in test_loader:
        for image, label in zip(images, labels):
            if len(wrong_images) > 21:
                break

            pred = model(image.to(device)).argsort(1, descending=True).cpu().numpy()[0]
            label.item()

            if label.item() != pred[0]:
                wrong_images.append((image, label, pred))

    for i in range(len(wrong_images)):
        plt.subplot(5,7,i+1)
        image, label, pred = wrong_images[i]
        plt.imshow(image.numpy()[0], 'gray')
        plt.title(f"Predicted: {pred}, True: {label}")
        plt.axis('off')
    plt.show()

    path = 'figs/wrongpred.png'
    
    plt.savefig(path)
    return path


