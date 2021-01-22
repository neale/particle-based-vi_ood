import torch
import torch.nn as nn
import numpy as np
from scipy.stats import entropy as entropy_fn
from sklearn.metrics import roc_auc_score


def auc_score(known, unknown):
    """ Computes the AUROC for the given predictions on `known` data
        and `unknown` data.
    """
    y_true = np.array([0] * len(known) + [1] * len(unknown))
    y_score = np.concatenate([known, unknown])
    auc_score = roc_auc_score(y_true, y_score)
    return auc_score


def uncertainty(outputs): 
    """ outputs (torch.tensor): class probabilities, 
        in practice these are given by a softmax operation
        * Soft voting averages the probabilties across the ensemble
            dimension, and then takes the maximal predicted class
            Taking the entropy of the averaged probabilities does not 
            yield a valid probability distribution, but in practice its ok
    """
    # Soft Voting (entropy and var in confidence)
    if outputs.shape[0] > 1:
        preds_soft = outputs.mean(0)  # [data, dim]
    else:
        preds_soft = outputs[0, :, :]
    entropy = entropy_fn(preds_soft.T.cpu().numpy()) # [data]
    
    preds_hard = outputs.var(0).cpu()  # [data, dim]
    variance = preds_hard.sum(1).numpy()  # [data]
    return (entropy, variance)


class MattLoss(object):
    """ MattLoss
    """
    def __init__(self, do_softplus=True):
        self.softplus = nn.Softplus() if do_softplus else lambda x: x
    
    def kliep_loss(self, logits, labels, max_ratio=50):
        logits = torch.clamp(logits,min=-1*max_ratio, max=max_ratio)
        #preds  = torch.softmax(logits,dim=1)
        preds  = self.softplus(logits)
        #preds  = torch.sigmoid(logits) * 10
        maxlog = torch.log(torch.FloatTensor([max_ratio])).to(preds.device)
        y = torch.eye(preds.size(1))
        labels = y[labels].to(preds.device)
        inlier_loss  = (labels * (maxlog-torch.log(preds))).sum(1)
        outlier_loss = ((1-labels) * (preds)).mean(1)
        loss = (inlier_loss + outlier_loss).mean()#/preds.size(1)
        return loss
    
    def kliep_loss_sigmoid(self, logits, labels, max_ratio=10):
        preds  = torch.sigmoid(logits) * 10
        maxlog = torch.log(torch.FloatTensor([max_ratio])).to(preds.device)
        y = torch.eye(preds.size(1))
        labels = y[labels].to(preds.device)
        inlier_loss  = (labels * (maxlog-torch.log(preds))).sum(1)
        outlier_loss = ((1-labels) * (preds)).mean(1)
        loss = (inlier_loss + outlier_loss).mean()#/preds.size(1)
        return loss
    
    def ulsif_loss(self, logits, labels, max_ratio=50):
        logits = torch.clamp(logits,min=-1*max_ratio, max=max_ratio)
        #preds  = torch.softmax(logits,dim=1)
        preds  = self.softplus(logits)
        #preds  = torch.sigmoid(logits) * 10
        maxlog = torch.log(torch.FloatTensor([max_ratio])).to(preds.device)
        y = torch.eye(preds.size(1))
        labels = y[labels].to(preds.device)
        inlier_loss  = (labels * (-2*(preds))).sum(1)
        outlier_loss = ((1-labels) * (preds**2)).mean(1)
        loss = (inlier_loss + outlier_loss).mean()#/preds.size(1)
        return loss
    
    def power_loss(self, logits, labels, alpha=.1, max_ratio=50):
        logits = torch.clamp(logits,min=-1*max_ratio, max=max_ratio)
        preds  = self.softplus(logits)
        y = torch.eye(preds.size(1))
        labels = y[labels].to(preds.device)
        inlier_loss  = (labels * (1 - preds.pow(alpha))/(alpha)).sum(1)
        outlier_loss = ((1-labels) * (preds.pow(1+alpha)-1)/(1+alpha)).mean(1)
        loss = (inlier_loss + outlier_loss).mean()#/preds.size(1)
        return loss
    
    def power_loss_05(self, logits, labels): return self.power_loss(logits, labels, alpha=.05, max_ratio=50)
    def power_loss_10(self, logits, labels): return self.power_loss(logits, labels, alpha=.1, max_ratio=50)
    def power_loss_50(self, logits, labels): return self.power_loss(logits, labels, alpha=.5, max_ratio=50)
    def power_loss_90(self, logits, labels): return self.power_loss(logits, labels, alpha=.90, max_ratio=50)
    
    def get_loss_dict(self):
        return {
            'ce':nn.CrossEntropyLoss(),
            'kliep':   self.kliep_loss, 
            'ulsif':   self.ulsif_loss, 
            'power05': self.power_loss_05, 
            'power10': self.power_loss_10, 
            'power50': self.power_loss_50, 
            'power90': self.power_loss_90, 
        }
