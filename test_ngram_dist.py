from batch_all_triplet_loss import ngram_dist
import torch

a = torch.randn((10, 64))

dists = ngram_dist(a, 3, 1)
print(dists)
