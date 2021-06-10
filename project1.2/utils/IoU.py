import numpy as np
def IoU(mask1, mask2):
    intersection = np.sum((mask1.astype(int)+ mask2.astype(int)) == 2)
    
    union = np.sum(mask1) + np.sum(mask2) - intersection
    
    return intersection / union

