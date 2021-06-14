import torch
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

## IOU ##


## FOCAL LOSS ##

def focal_loss(y_real, y_pred):
    gamma = 2
    y_pred = torch.clamp(torch.sigmoid(y_pred), 1e-8, 1-1e-8)
    return -torch.mean((1-y_pred)**gamma * y_real *torch.log(y_pred) + (1-y_real)*torch.log(1 -y_pred))

## ACCURACY ##

## SENSITIVITY ##

## SPECIFICITY ##

## PLOT IMAGES ##

def plotimages(dataloader, model):
    images, labels = next(iter(dataloader))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output, _ = model(images.to(device))

    plt.figure(figsize=(20,10))
    for i in range(0, 21, 3):
        plt.subplot(5,7,i+1)
        plt.imshow(images[i].numpy()[0], 'gray')
        plt.title('Image')
        plt.axis('off')

        plt.subplot(5,7,i+2)
        plt.imshow(labels[i].numpy()[0], 'gray')
        plt.title('Label')
        plt.axis('off')

        plt.subplot(5,7,i+3)
        plt.imshow(output[i].detach().cpu().numpy()[0], 'gray')
        plt.title('Prediction')
        plt.axis('off')
    plt.show()
    path = 'figs/lung.png'
    plt.savefig(path)
    
    return path
