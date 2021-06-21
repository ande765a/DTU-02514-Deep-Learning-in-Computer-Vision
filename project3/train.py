import os
import wandb
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, dataloader
from utils import sample_images, imgBuffer

def train(model_G, model_D, optimizer, train_set, test_set, num_workers=8, num_epochs=10, batch_size=64, loss_funcs={"GAN":nn.BCELoss, "Cycle": nn.L1Loss, "Identity": nn.L1Loss}):
    wandb.init(project='horse2zebra', entity='dlincv')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    A_hat_buffer = imgBuffer()
    B_hat_buffer = imgBuffer()

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    GANLoss = loss_funcs["GAN"]()
    cycleLoss = loss_funcs["Cycle"]()
    identityLoss = loss_funcs["Identity"]()

    config = wandb.config
    config.batch_size = batch_size
    config.epochs = num_epochs
    config.optimizer = optimizer
    config.loss_func = loss_funcs["GAN"]

    img_shape = next(iter(train_loader))[0][0].shape

    # Setup Generator for A to B and discriminator for B
    G_a2b = model_G().to(device)
    D_b = model_D(tuple(img_shape)).to(device)
    wandb.watch(G_a2b)
    wandb.watch(D_b)

    # Setup Generator for B to A and discriminator for A
    G_b2a = model_G().to(device)
    D_a = model_D(tuple(img_shape)).to(device)
    wandb.watch(D_a)
    wandb.watch(G_b2a)

    lambda_cycle = 10
    lambda_identity = 5

    d_params = list(D_a.parameters()) + list(D_b.parameters())
    g_params = list(G_a2b.parameters()) + list(G_b2a.parameters())
    
    d_opt = optimizer(d_params, 0.0002, (0.5, 0.999))
    g_opt = optimizer(g_params, 0.0002, (0.5, 0.999))


    for epoch in tqdm(range(num_epochs), unit='epoch'):
        G_a2b.train()
        D_b.train()
        
        G_b2a.train()
        D_a.train()

        loss_GAN_total_epoch = []
        loss_GAN_epoch = []
        loss_cycle_epoch = []
        loss_identity_epoch = []
        
        # loss_d_a_epoch = []
        # loss_d_b_epoch = []
        loss_d_epoch = []
        # Train loop
        for A, B in tqdm(train_loader):
            A = A.to(device)
            B = B.to(device)
            real_tensor = torch.ones((A.shape[0], *D_a.output_shape)).to(device)
            fake_tensor = torch.zeros((A.shape[0], *D_a.output_shape)).to(device)
            
            d_opt.zero_grad()
            g_opt.zero_grad()

            A_hat = G_b2a(B)
            B_hat = G_a2b(A)

            A_hat_pred = D_a(A_hat.detach())
            A_pred = D_a(A)
            B_hat_pred = D_b(B_hat.detach())
            B_pred = D_b(B)
            

            # Generator loss
            loss_a2b_generator = GANLoss(B_hat_pred, real_tensor)
            loss_b2a_generator = GANLoss(A_hat_pred, real_tensor)
            
            loss_cycle_consistency_A = cycleLoss(G_a2b(A_hat), B)
            loss_cycle_consistency_B = cycleLoss(G_b2a(B_hat), A)
           
            # Identity loss            
            loss_identity_A = identityLoss(G_b2a(A), A)
            loss_identity_B = identityLoss(G_a2b(B), B)

            # Combining generator losses
            loss_cycle = lambda_cycle*(loss_cycle_consistency_A + loss_cycle_consistency_B)
            loss_identity = lambda_identity*(loss_identity_A + loss_identity_B)
            loss_GAN = loss_a2b_generator + loss_b2a_generator

            loss_g = loss_GAN + loss_cycle + loss_identity
                  
            loss_g.backward()
            g_opt.step()

            # Discriminator losses
            loss_a_real_discriminator = GANLoss(A_pred, real_tensor)
            loss_b_real_discriminator = GANLoss(B_pred, real_tensor)

            A_hat_ = A_hat_buffer.add_and_sample(A_hat)
            loss_a_fake_discriminator = GANLoss(D_a(A_hat_.detach()), fake_tensor)

            B_hat_ = B_hat_buffer.add_and_sample(B_hat)
            loss_b_fake_discriminator = GANLoss(D_b(B_hat_.detach()), fake_tensor)

            # Combining generator losses
            loss_d_a = loss_a_real_discriminator + loss_a_fake_discriminator 
            loss_d_b = loss_b_real_discriminator + loss_b_fake_discriminator
            

            loss_d = loss_d_a + loss_d_b
            loss_d.backward()
            d_opt.step()

            loss_GAN_total_epoch.append(loss_g.item())
            loss_GAN_epoch.append(loss_GAN.item())
            loss_cycle_epoch.append(loss_cycle.item())
            loss_identity_epoch.append(loss_identity.item())
            
            # loss_d_a_epoch.append(loss_d_a)
            # loss_d_b_epoch.append(loss_d_b)
            loss_d_epoch.append(loss_d.item())

        # Validation loop
        test_loss_epoch = []

        G_a2b.eval()
        G_a2b.eval()
        D_a.eval()
        D_b.eval()
        
        test_loss_GAN_total_epoch = []
        test_loss_D_epoch = []
        # Test loop
        for A, B in tqdm(test_loader):
            A = A.to(device)
            B = B.to(device)
            real_tensor = torch.ones((A.shape[0], *D_a.output_shape)).to(device)
            fake_tensor = torch.zeros((A.shape[0], *D_a.output_shape)).to(device)
            A_hat = G_b2a(B)
            B_hat = G_a2b(A)

            A_hat_pred = D_a(A_hat.detach())
            A_pred = D_a(A)
            B_hat_pred = D_b(B_hat.detach())
            B_pred = D_b(B)
            

            # Generator loss
            loss_a2b_generator = GANLoss(B_hat_pred, real_tensor)
            loss_b2a_generator = GANLoss(A_hat_pred, real_tensor)
            
            loss_cycle_consistency_A = cycleLoss(G_a2b(A_hat), A)
            loss_cycle_consistency_B = cycleLoss(G_b2a(B_hat), B)
           
            # Identity loss            
            loss_identity_A = identityLoss(G_b2a(A), A)
            loss_identity_B = identityLoss(G_a2b(B), B)

            #Combining generator losses
            loss_cycle = lambda_cycle*(loss_cycle_consistency_A + loss_cycle_consistency_B)
            loss_identity = lambda_identity*(loss_identity_A + loss_identity_B)
            loss_GAN = loss_a2b_generator + loss_b2a_generator

            loss_g = loss_GAN + loss_cycle + loss_identity
                  

            # Discriminator losses
            loss_a_real_discriminator = GANLoss(A_pred, real_tensor)
            loss_b_real_discriminator = GANLoss(B_pred, real_tensor)

            # TODO: use buffer here?
            loss_a_fake_discriminator = GANLoss(A_hat_pred, fake_tensor)
            loss_b_fake_discriminator = GANLoss(B_hat_pred, fake_tensor)

            # Combining generator losses
            loss_d_a = loss_a_real_discriminator + loss_a_fake_discriminator 
            loss_d_b = loss_b_real_discriminator + loss_b_fake_discriminator
            

            loss_d = loss_d_a + loss_d_b

            test_loss_GAN_total_epoch.append(loss_g.item())
            test_loss_D_epoch.append(loss_d.item())

        loss_generator_total_mean = np.mean(loss_GAN_total_epoch)
        loss_generator_mean = np.mean(loss_GAN_epoch)
        loss_cycle_mean = np.mean(loss_cycle_epoch)
        loss_identity_mean = np.mean(loss_identity_epoch)
        # loss_D_A_mean = torch.mean(loss_d_a_epoch)
        # loss_D_B_mean = torch.mean(loss_d_b_epoch)
        loss_D_mean = np.mean(loss_d_epoch)

        test_loss_generator_total_mean = np.mean(test_loss_GAN_total_epoch)
        test_loss_D_mean = np.mean(test_loss_D_epoch)
        print(f"Train loss GAN: {loss_generator_total_mean:.3f}\t Test loss GAN: {test_loss_generator_total_mean:.3f}\t",
              f"Train loss discriminator: {loss_D_mean:.3f}\t Test loss discriminator: {test_loss_D_mean :.3f}")
        
        wandb.log({ 
            "loss_Generator_total": loss_generator_total_mean,
            "loss_Generator": loss_generator_mean,
            "loss_cycle": loss_cycle_mean,
            "loss_identity": loss_identity_mean,
            "loss_Discriminator": loss_D_mean,
            "test_loss_Generator_total": test_loss_generator_total_mean,
            "test_loss_Discriminator": test_loss_D_mean
        })


        img_path = sample_images(test_loader, G_a2b, G_b2a, epoch)
        wandb.save(img_path)
    

    # Save models
    weights_path = os.path.join(wandb.run.dir,"G_a2b.pth")
    torch.save(G_a2b.state_dict(),weights_path)
    wandb.save(weights_path)
    weights_path = os.path.join(wandb.run.dir,"G_b2a.pth")
    torch.save(G_b2a.state_dict(),weights_path)
    wandb.save(weights_path)
    weights_path = os.path.join(wandb.run.dir,"D_a.pth")
    torch.save(D_a.state_dict(),weights_path)
    wandb.save(weights_path)
    weights_path = os.path.join(wandb.run.dir,"D_b.pth")
    torch.save(D_b.state_dict(),weights_path)
    wandb.save(weights_path)
    print(f'Path to model is: {wandb.run.dir}')
    
    return 
