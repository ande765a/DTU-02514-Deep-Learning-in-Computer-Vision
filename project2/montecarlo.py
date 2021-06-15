import torch

def montecarlo(model, image, num_loop = 1000):
    pred = torch.zeros(num_loop, image.shape[0], image.shape[1])
    for i in num_loop:
        temp, _ = model(image) # Predict image, while dropout true
        pred[i,:,:] = temp[0]/num_loop #Weighted by 1/num_loop
    return torch.sum(pred, axis = 0)
