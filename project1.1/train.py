import numpy as np
from tqdm import tqdm
import wandb
import sys
from medcam import medcam
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt  

def checkpoint(model):
    torch.save(model.state_dict(), wandb.run.dir + wandb.run.name + '.pth')
    wandb.save(wandb.run.dir + wandb.run.name + '.pth')


#We define the training as a function so we can easily re-use it.
def train(model, optimizer, trainset, testset, config, num_epochs=10, batch_size=64, save_weights=False, patience=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    # Loss function
    criterion = nn.BCELoss()
    def loss_fun(output, target):
        return criterion(output.to(torch.float), target.to(torch.float))
    out_dict = {'train_acc': [],
              'test_acc': [],
              'train_loss': [],
              'test_loss': []}  
    finish_epoch = num_epochs
    atl_test_loss = sys.maxsize
    countdown = patience
    for epoch in tqdm(range(num_epochs), unit='epoch'):
        model.train()
        #For each epoch
        train_correct = 0
        train_loss = []
        for minibatch_no, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
            data, target = data.to(device), target.to(device)
            #Zero the gradients computed for each weight
            optimizer.zero_grad()
            #Forward pass your image through the network
            output = model(data)
            #Compute the loss
            loss = loss_fun(output, target)
            #Backward pass through the network
            loss.backward()
            #Update the weights
            optimizer.step()

            #Compute how many were correctly classified
            train_loss.append(loss.item())
            predicted = (output > 0.5).to(torch.int)
            train_correct += (target==predicted).sum().cpu().item()

        #Comput the test accuracy
        test_loss = []
        test_correct = 0
        model.eval()
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            with torch.no_grad():
                output = model(data)
            test_loss.append(loss_fun(output, target).cpu().item())
            predicted = (output > 0.5).to(torch.int)
            
            test_correct += (target==predicted).sum().cpu().item()
        
        out_dict['train_acc'].append(train_correct/len(trainset))
        out_dict['test_acc'].append(test_correct/len(testset))
        out_dict['train_loss'].append(np.mean(train_loss))
        out_dict['test_loss'].append(np.mean(test_loss))
        print(f"Loss train: {np.mean(train_loss):.3f}\t test: {np.mean(test_loss):.3f}\t",
              f"Accuracy train: {out_dict['train_acc'][-1]*100:.1f}%\t test: {out_dict['test_acc'][-1]*100:.1f}%")
        
        wandb.log({"train_acc": out_dict['train_acc'][-1]})
        wandb.log({"test_acc": out_dict['test_acc'][-1]})
        wandb.log({"train_loss": out_dict['train_loss'][-1]})
        wandb.log({"test_loss": out_dict['test_loss'][-1]})
        
        wandb.log({"train_acc": out_dict['train_acc'][-1], "test_acc": out_dict['test_acc'][-1]})
        wandb.log({"train_loss": out_dict['train_loss'][-1], "test_loss": out_dict['test_loss'][-1]})

        current_test_loss = out_dict['test_loss'][-1]
        
        if atl_test_loss > current_test_loss:
            atl_test_loss = current_test_loss
            countdown = patience
        else:
            countdown -= 1
            if countdown == 0:
                finish_epoch = epoch
                print(f'Stopping at epoch {epoch} because test loss haven\'t improved in the last {patience} epochs')
                break

        if save_weights and atl_test_loss > current_test_loss:
            checkpoint(model)
            
    config.epochs_run = finish_epoch
    
    # ###PLOTTING###
    # model_ = medcam.inject(model, output_dir="attention_maps", save_maps=True)
    # model_.eval()

    # for batch in next(iter(test_loader)):
    #     output = model_(batch.to(device))
    #     print(output)

    return out_dict
