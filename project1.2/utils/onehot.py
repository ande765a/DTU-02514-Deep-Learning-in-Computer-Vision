import torch

def onehot(k, n):
	t = torch.zeros(k.shape[0], n)
	t[torch.arange(k.shape[0]), k] = 1.
	t = t.long()
	return t