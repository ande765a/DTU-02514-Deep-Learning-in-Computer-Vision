import os
import torch
import wandb
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

## FP, FN, TP, TN ##
def confusionmatrix(pred, true):
    ## returns: TN, FP, FN, TTP
    tn, fp, fn, tp = confusion_matrix(true,pred).ravel()
    return tn, fp, fn, tp


## BCE LOSS ##
def bce_loss(y_real, y_pred):
    return torch.mean(y_pred - y_real*y_pred + torch.log(1 + torch.exp(-y_pred)))


## DICE METRICS ##
def dice_coef(y_pred, y_real):
    y_r_f = y_real
    y_p_f = y_pred
    
    intersection = torch.mean(y_r_f * y_p_f * 2 + 1)
    union = torch.mean(y_r_f + y_p_f + 1)
    return intersection / union


def dice_loss(y_real, y_pred):
    dice_co = dice_coef(y_real, y_pred)
    return 1 - dice_co


## FOCAL LOSS ##
def focal_loss(y_real, y_pred):
    gamma = 2
    y_pred = torch.clamp(torch.sigmoid(y_pred), 1e-8, 1-1e-8)
    return -torch.mean((1-y_pred)**gamma * y_real *torch.log(y_pred) + (1-y_real)*torch.log(1 -y_pred))


## PLOT IMAGES ##
def plotimages(dataloader, model, figName, figPath='figs/'):
    images, *labels = next(iter(dataloader))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output, _ = model(images.to(device))
    height, width = 6, 3

    fig, ax = plt.subplots(height, width, figsize=(10, 30))
    for i in range(0, height):
        ax[i, 0].imshow(images[i].numpy()[0], "gray")
        ax[i, 0].axis("off")

        ax[i, 1].imshow(labels[0][i].numpy()[0], "gray")
        ax[i, 1].axis("off")

        ax[i, 2].imshow(output[i].detach().cpu().numpy()[0], "gray")
        ax[i, 2].axis("off")
    fig.tight_layout()
    path = os.path.join(figPath, figName)
    fig.savefig(path)
    
    return path
    

def measures(TP, TN, FP, FN):
        accuracy = (1 + TP + TN) / (TP + TN + FP +FN +1)
        precision = (TP +1) / (TP + FP +1)
        recall = (TP + 1) / (TP + FN + 1)
        dice = 2/ ((1/precision) + (1/recall))
        specificity = (TN +1 ) / (TN + FP +1)
        sensitivity = (TP +1) / (TP + FN +1)
        iou = (TP +1) / (TP + FP + FN +1)
        return accuracy, dice, specificity, sensitivity, iou


def run_test(model, test_loader, criterion, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    TP_val = 0 
    FP_val = 0 
    TN_val = 0 
    FN_val = 0
    test_loss = []
    
    model.eval()
    for data, *targets in test_loader:
        for target in targets:
            data, target = data.to(device), target.to(device)
            output, logits = model(data)
            
            loss = criterion(logits, target).cpu().item()
            
            test_loss.append(loss)
                        
            output = torch.where(output > 0.5, 1, 0)
            
            TP_val += torch.sum(torch.where((target == 1) & (output == 1), 1, 0))
            TN_val += torch.sum(torch.where((target == 0) & (output == 0), 1, 0))
            FP_val += torch.sum(torch.where((target == 0) & (output == 1), 1, 0))
            FN_val += torch.sum(torch.where((target == 1) & (output == 0), 1, 0))

    accuracy_test, dice_test, specificity_test, sensistivity_test, iou_test = measures(TP_val, TN_val, FP_val, FN_val)

    config.test_accuracy = accuracy_test
    config.test_loss = np.mean(test_loss)
    config.test_dice = dice_test
    config.test_specificity = specificity_test
    config.test_sensitivity = sensistivity_test
    config.test_iou = iou_test

    img_path = plotimages(test_loader, model, 'lung_test.png')
    wandb.save(img_path)


class WeightedFocalLoss(torch.nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, alpha=.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        # print(inputs.shape, targets.shape)
        # at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)

        # print(pt.shape, BCE_loss.shape)
        # F_loss = at*(1-pt)**self.gamma * BCE_loss
        F_loss = (1-pt)**self.gamma * BCE_loss

        return F_loss.mean()


def iou(mask1, mask2):
    intersection = torch.sum(mask1 & mask2, dim=list(range(1, len(mask1.shape))))
    union = torch.sum(mask1 | mask2, dim=list(range(1, len(mask1.shape))))
    return intersection.true_divide(union + 1e-8)

def distance(y, y_hat):
    return 1 - iou(y, y_hat)

def generalized_energy_distance(models, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        term1_distances = []
        for image, *annotations in test_loader:
            model = np.random.choice(models)
            y = np.random.choice(annotations)
            y_hat, _ = model(image.to(device))            
            y_hat = y_hat > 0.5
            d = distance((y==1).to(device), y_hat)
            term1_distances.append(d)
        print('Done with first term')
        term2_distances = []
        for image, *annotations in test_loader:
            model = np.random.choice(models)
            y, y_prime = np.random.choice(annotations, replace=True, size=2)
            d = distance(y==1, y_prime==1)
            term2_distances.append(d)
        print('Done with second term')

        term3_distances = []
        for image, *annotations in test_loader:
            model, model_prime = np.random.choice(models, replace=True, size=2)
            (y_hat, _), (y_hat_prime, _) = model(image.to(device)), model_prime(image.to(device))
            y_hat = y_hat > 0.5
            y_hat_prime = y_hat_prime > 0.5
            d = distance(y_hat, y_hat_prime)
            term3_distances.append(d)
        print('Done with third term')

        return 2 * torch.cat(term1_distances).mean() - torch.cat(term2_distances).mean() - torch.cat(term3_distances).mean()