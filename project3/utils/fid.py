import torch
import numpy as np
from tqdm import tqdm
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance

# Code is inspired by: https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py

def calculate_fids(test_loader, G_a2b, G_b2a):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    inception_v3_model = InceptionV3([block_idx]).to(device)

    G_a2b.eval()
    G_b2a.eval()
    inception_v3_model.eval()

    acts_A = []
    acts_gen_A = []
    acts_B = []
    acts_gen_B = []

    for A, B in tqdm(test_loader):
        A = A.to(device)
        B = B.to(device)

        gen_A = G_b2a(B)
        gen_B = G_a2b(A)

        with torch.no_grad():
            act_A = inception_v3_model(A)[0].squeeze(3).squeeze(2).cpu().numpy()
            act_gen_A = inception_v3_model(gen_A)[0].squeeze(3).squeeze(2).cpu().numpy()
            act_B = inception_v3_model(B)[0].squeeze(3).squeeze(2).cpu().numpy()
            act_gen_B = inception_v3_model(gen_B)[0].squeeze(3).squeeze(2).cpu().numpy()

            acts_A.append(act_A)
            acts_gen_A.append(act_gen_A)
            acts_B.append(act_B)
            acts_gen_B.append(act_gen_B)

    acts_A = np.hstack(acts_A)
    acts_gen_A = np.hstack(acts_gen_A)
    acts_B = np.hstack(acts_B)
    acts_gen_B = np.hstack(acts_gen_B)


    mu_A = np.mean(acts_A, axis=0)
    sigma_A = np.cov(acts_A, rowvar=False)

    mu_gen_A = np.mean(acts_gen_A, axis=0)
    sigma_gen_A = np.cov(acts_gen_A, rowvar=False)

    mu_B = np.mean(acts_B, axis=0)
    sigma_B = np.cov(acts_B, rowvar=False)

    mu_gen_B = np.mean(acts_gen_B, axis=0)
    sigma_gen_B = np.cov(acts_gen_B, rowvar=False)

    
    fid_A = calculate_frechet_distance(mu_A, sigma_A, mu_gen_A, sigma_gen_A)
    fid_B = calculate_frechet_distance(mu_B, sigma_B, mu_gen_B, sigma_gen_B)

    return fid_A, fid_B