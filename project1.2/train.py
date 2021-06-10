import numpy as np
from tqdm import tqdm
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train(model, optimizer, trainset, testset, config, num_epochs=10, batch_size=64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    criterion = nn.BCELoss()

    train_acc = []
    test_acc = []
    train_loss = []
    test_loss = []

    for epoch in tqdm(range(num_epochs), unit='epoch'):
        
        # Train loop
        train_loss = []
        train_correct = 0
        model.train()
        for data, target in tqdm(train_loader):
            optimizer.zero_grad()


            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

            predicted = (output > 0.5).to(torch.int)
            train_correct += (target == predicted).sum().cpu().item()


        # Test loop
        test_loss = []
        test_correct = 0
        model.eval()
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
      
            output = model(data)
            loss = criterion(output, target).cpu().item()
            test_loss.append(loss)

            predicted = (output > 0.5).to(torch.int)
            test_correct += (target == predicted).sum().cpu().item()
        
        train_acc.append(train_correct/len(trainset))
        test_acc.append(test_correct/len(testset))
        train_loss.append(np.mean(train_loss))
        test_loss.append(np.mean(test_loss))


        print(f"Loss train: {np.mean(train_loss):.3f}\t test: {np.mean(test_loss):.3f}\t",
              f"Accuracy train: {train_acc[-1] * 100:.1f}%\t test: {test_acc[-1] * 100:.1f}%")
        
        wandb.log({ 
            "train_acc": train_acc[-1],
            "test_acc": test_acc[-1],
            "train_loss": train_loss[-1],
            "test_loss": test_loss[-1]
        })
        
    # Save model
    torch.save(model.state_dict(), wandb.run.dir + wandb.run.name + '.pth')
    wandb.save(wandb.run.dir + wandb.run.name + '.pth')

    return train_acc, test_acc, train_loss, test_loss
