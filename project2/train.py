import numpy as np
from tqdm import tqdm
import wandb
import torch
import torch.nn as nn
from utils import bce_loss, dice_loss, focal_loss, confusionmatrix, dice_coef, plotimages, measures
from torch.utils.data import DataLoader


def train(model, optimizer, train_set, validation_set, config, num_workers=8, num_epochs=10, batch_size=64):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    criterion = nn.BCEWithLogitsLoss()
    img_size = 128
    train_acc = []
    validation_acc = []
    train_loss = []
    validation_loss = []
    dice_score_val = []
    dice_score_train = []
    for epoch in tqdm(range(num_epochs), unit='epoch'):
        
        # Train loop
        train_total = 0
        train_loss_epoch = []
        train_correct = 0
        TP_train = 0 
        TN_train = 0 
        FN_train = 0 
        FP_train = 0
        model.train()
        for data, target in tqdm(train_loader):
            optimizer.zero_grad()
            data, target = data.to(device), target.to(device)
        
            output, logits = model(data)
            output = torch.where(output > 0.5, 1, 0)

            loss = criterion(logits, target)
            
            loss.backward()
            optimizer.step()
            train_loss_epoch.append(loss.item())

            predicted = output
            
            TP_train += torch.sum(torch.where((target == 1) & (output == 1), 1, 0))
            TN_train += torch.sum(torch.where((target == 0) & (output == 0), 1, 0))
            FP_train += torch.sum(torch.where((target == 0) & (output == 1), 1, 0))
            FN_train += torch.sum(torch.where((target == 1) & (output == 0), 1, 0))

        accuracy_train, dice_train, specificity_train, sensistivity_train = measures(TP_train, TN_train, FP_train, FN_train)
        # Validation loop
        val_loss_epoch = []
        val_correct = 0
        model.eval()
        val_total = 0 
        
        TP_val = 0 
        FP_val = 0 
        TN_val = 0 
        FN_val = 0
        for data, target in validation_loader:
            num_classes = target.shape[3]
            data, target = data.to(device), target.to(device)
            output, logits = model(data)#.view(data.shape[0], -1)
            
            loss = criterion(logits, target).cpu().item()
            
            val_loss_epoch.append(loss)
                        
            output = torch.where(output > 0.5, 1, 0)
            
            TP_val += torch.sum(torch.where((target == 1) & (output == 1), 1, 0))
            TN_val += torch.sum(torch.where((target == 0) & (output == 0), 1, 0))
            FP_val += torch.sum(torch.where((target == 0) & (output == 1), 1, 0))
            FN_val += torch.sum(torch.where((target == 1) & (output == 0), 1, 0))
            
        accuracy_val, dice_val, specificity_val, sensistivity_val = measures(TP_val, TN_val, FP_val, FN_val)
        
        #print(f'Dice is {dice_val},\n accuracy: {accuracy_val},\n specificity {specificity_val},\n sensitivity {sensistivity_val}')
        #print(f'{dice_score}')
        
        # print(f'val correct {val_correct}')
        train_acc.append(accuracy_train)
        validation_acc.append(accuracy_val)
        train_loss.append(np.mean(train_loss_epoch))
        validation_loss.append(np.mean(val_loss_epoch))
        dice_score_train.append(dice_train)
        dice_score_val.append(dice_val)

        print(f"Loss train: {np.mean(train_loss_epoch):.3f}\t validation: {np.mean(val_loss_epoch):.3f}\t",
              f"Accuracy train: {train_acc[-1] * 100:.1f}%\t validation: {validation_acc[-1] * 100:.1f}%")
        
        wandb.log({ 
            "train_acc": train_acc[-1],
            "validation_acc": validation_acc[-1],
            "train_loss": train_loss[-1],
            "validation_loss": validation_loss[-1],
            "train_dice_score": dice_score_train[-1],
            "val_dice_score": dice_score_val[-1]
        })
    img_path = plotimages(validation_loader, model, 'lungs_validation.png')
    wandb.save(img_path)
    # Save model
    torch.save(model.state_dict(), wandb.run.dir + wandb.run.name + '.pth')
    wandb.save(wandb.run.dir + wandb.run.name + '.pth')
    print(f'Path to model is: {wandb.run.dir + wandb.run.name + ".pth"}')
    
    return train_acc, validation_acc, train_loss, validation_loss
