from torchvision.models import inception_v3

def LSGAN_d_loss(x_real_pred, x_fake_pred, a = -1, b = 1):
    return 0.5*((x_real_pred -b)**2).mean() + 0.5*((x_fake_pred - a)**2).mean()

def LSGAN_g_loss(x_fake_pred, c = 0):
    return 0.5 * ((x_fake_pred - c)**2).mean()