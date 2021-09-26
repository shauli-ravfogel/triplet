import torch
import numpy as np
import copy
import torch.nn.functional as F
from torch import nn
import random


CUDA = False

class HardNegativeSampler(object):

    def __init__(self, k = 5):

        self.k = k

    def _get_mask(self, labels, positive = True):

        diffs =  labels[None, :] - (labels[None, :]).T

        if positive:

            mask = diffs == 0

        else:

            mask = diffs != 0

        if positive:
            mask[range(len(mask)), range(len(mask))] = 0
        return mask

    def get_distances(self, labels, dists):

        mask_anchor_positive = self._get_mask(labels, positive = True)
        mask_anchor_negative = self._get_mask(labels, positive = False)

        anchor_positive_dist = mask_anchor_positive * dists
        hardest_positive_idx = np.argmax(anchor_positive_dist, axis=1)
        max_anchor_negative_dist = np.max(dists, axis=1, keepdims=True)

        anchor_negative_dist = dists + max_anchor_negative_dist * (1 - mask_anchor_negative)
        k = int(np.random.choice(range(1, self.k + 1)))
        #k=0
        hardest_negatives_idx = np.argpartition(anchor_negative_dist, k, axis = 1)[:,k]
        hardest_negatives_idx2 = np.argmin(anchor_negative_dist, axis = 1)

        return hardest_positive_idx, hardest_negatives_idx

        #return dists[range(len(dists)), hardest_positive_idx], dists[range(len(dists)), hardest_negatives_idx]


def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    dist[dist != dist] = 0  # replace nan values with 0
    return torch.clamp(dist, 0.0, np.inf)



class BatchHardTripletLoss(torch.nn.Module):

    def __init__(self, p = 2, alpha = 0.1, normalize = False, mode = "euc", final = "plus", k = 1, do_max = True):

        super(BatchHardTripletLoss, self).__init__()
        self.p = p
        self.alpha = alpha
        self.normalize = normalize
        self.mode = mode
        self.final = final
        self.sampler = HardNegativeSampler(k = k)
        self.k = k
        self.do_max = do_max
        
    def get_mask(self, labels, positive = True):

        diffs =  labels[None, :] - torch.t(labels[None, :])

        if positive:

            mask = diffs == 0

        else:

            mask = diffs != 0

        if positive:
            mask[range(len(mask)), range(len(mask))] = 0
        return mask

    def forward(self, H, labels):

        if self.normalize or self.mode == "cosine":

            H = H / torch.norm(H, dim = 1, p = self.p, keepdim = True)

        if self.mode == "euc":
            dists = pairwise_distances(H)
        elif self.mode == "cosine":
            dists = 1. - H @ torch.t(H)

        dists = torch.clamp(dists, min = 1e-7)

        hardest_positive_idx, hardest_negatives_idx = self.sampler.get_distances(labels.detach().cpu().numpy(), dists.detach().cpu().numpy())
        hardest_positive_idx, hardest_negatives_idx = torch.tensor(hardest_positive_idx), torch.tensor(hardest_negatives_idx)
        if CUDA:
            hardest_positive_idx, hardest_negatives_idx = hardest_positive_idx.cuda(), hardest_negatives_idx.cuda()     

        hardest_negative_dist = dists.gather(1, hardest_negatives_idx.view(-1,1))
        hardest_positive_dist = dists.gather(1, hardest_positive_idx.view(-1,1))

        differences = hardest_positive_dist - hardest_negative_dist

        if self.final == "plus":

            if self.do_max:
                
                triplet_loss = torch.max(differences + self.alpha, torch.zeros_like(differences))
            else:
                triplet_loss = differences# + self.alpha
        elif self.final == "softplus":
            triplet_loss = F.softplus(differences)
        elif self.final == "softmax":
            temp = 1 if self.mode != "cosine" else 1
            z = torch.max(hardest_positive_dist, hardest_negative_dist)
            pos = torch.exp((hardest_positive_dist - z)/temp)
            neg = torch.exp((hardest_negative_dist - z)/temp)
            triplet_loss = (pos / (pos + neg))**2
        else:
            triplet_loss = hardest_positive_dist - hardest_negative_dist


        relevant = triplet_loss[triplet_loss > 1e-7]
        good = (differences < 0).sum() 
        bad = H.shape[0] - good

        return torch.mean(relevant)#, torch.mean(differences), good, bad
