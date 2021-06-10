from torch.utils.data import DataLoader
from plotimage import plotimages

def analyze_data(trainset, batch_size):
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    mean = 0.
    std = 0.
    for images, _ in train_loader:
        batch_samples = images.size(0) # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= len(train_loader.dataset)
    std /= len(train_loader.dataset)  
    
    # print(mean, std)

    filename = plotimages(train_loader)
    
    return filename
    