import os
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
# def

## FOCAL LOSS ##

def focal_loss(y_real, y_pred):
    gamma = 2
    y_pred = torch.clamp(torch.sigmoid(y_pred), 1e-8, 1-1e-8)
    return -torch.mean((1-y_pred)**gamma * y_real *torch.log(y_pred) + (1-y_real)*torch.log(1 -y_pred))



## PLOT IMAGES ##

def plotimages(dataloader, model, figName, figPath='figs/'):
    images, labels = next(iter(dataloader))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output, _ = model(images.to(device))

    plt.figure(figsize=(40,40))
    fig, ax = plt.subplots(10,3)
    for i in range(0, 10):
        ax[i, 0].imshow(images[i].numpy()[0], "gray")
        # ax[i, 0].set_title("Image")
        ax[i, 0].axis("off")

        ax[i, 1].imshow(labels[i].numpy()[0], "gray")
        # ax[i, 1].set_title("Label")
        ax[i, 1].axis("off")

        ax[i, 2].imshow(output[i].detach().cpu().numpy()[0], "gray")
        # ax[i, 2].set_title("Prediction")
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
        return accuracy, dice, specificity, sensitivity

def run_test(model, test_loader, criterion):
    pass


class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, alpha=.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()