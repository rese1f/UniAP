import torch
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
import math
import numpy as np

from dataset.UniASET_constants import TASKS, TASKS_CLS



CLS_IDXS = [TASKS.index(task) for task in TASKS_CLS]


# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss

    Params:
        num: int,the number of loss
        x: multi-task loss
    Examples:
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        # acc = (TP + TN) / (TP + TN + FP + TN)
        Acc = np.diag(self.confusion_matrix).sum() / \
            self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        # acc = (TP) / TP + FP
        Acc = np.diag(self.confusion_matrix) / \
            self.confusion_matrix.sum(axis=1)
        Acc_class = np.nanmean(Acc)
        return Acc_class

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
            np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusion_matrix, axis=1) / \
            np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
            np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

def measure_pa_miou(num_class, gt_image, pre_image):
        metric = Evaluator(num_class)
        metric.add_batch(gt_image, pre_image)
        acc = metric.Pixel_Accuracy()
        mIoU = metric.Mean_Intersection_over_Union()
        
        return acc, mIoU      


def generate_CLS_mask(t_idx):
    '''
    Generate binary mask.
    '''
    cls_mask = torch.zeros_like(t_idx, dtype=bool)
    for cls_idx in CLS_IDXS:
        cls_mask = torch.logical_or(cls_mask, t_idx == cls_idx)

    return cls_mask

def adaptive_loss(Y_src, Y_tgt, M, t_idx):
    '''
    Compute l1 loss for continuous tasks and bce loss for semantic segmentation.
    [loss_args]
        Y_src: unnormalized prediction of shape (B, T, N, 1, H, W)
        Y_tgt: normalized GT of shape (B, T, N, 1, H, W)
        M    : mask for loss computation of shape (B, T, N, 1, H, W)
        t_idx: task index of shape (B, T)
    '''

    # prediction loss
    B, T, N, C, H, W = Y_src.shape
    device = Y_src.device
    loss_bce = F.binary_cross_entropy_with_logits(Y_src, Y_tgt, reduction='none')
    loss_l1 = F.l1_loss(Y_src.sigmoid(), Y_tgt, reduction='none')


    loss = loss_bce + loss_l1
    
    loss_bce = rearrange(loss_bce, 'B T N C H W -> B (T N C H W)')
    loss_bce = torch.mean(loss_bce, dim=1)
    loss_l1 = rearrange(loss_l1, 'B T N C H W -> B (T N C H W)')
    loss_l1 = torch.mean(loss_l1, dim=1)   
    
    loss = rearrange(loss, 'B T N C H W -> B (T N C H W)')
    loss = torch.mean(loss, dim=1)
    t_idx = rearrange(t_idx, 'B T -> (B T)')

    # task split
    PE_mask = torch.zeros_like(t_idx, dtype=torch.float)
    SS_mask = torch.zeros_like(t_idx, dtype=torch.float)
    CLS_mask = torch.zeros_like(t_idx, dtype=torch.float)
    PE_mask = torch.where(t_idx==0, torch.tensor([1], device=device), PE_mask)
    SS_mask = torch.where(t_idx==1, torch.tensor([1], device=device), SS_mask)
    CLS_mask = torch.where(t_idx==2, torch.tensor([1], device=device), CLS_mask)

  
    # adaptive loss weight
    N_PE =  torch.sum(PE_mask) if torch.sum(PE_mask) != 0 else 1
    N_SS = torch.sum(SS_mask) if torch.sum(SS_mask) != 0 else 1
    N_CLS = torch.sum(CLS_mask) if torch.sum(CLS_mask) != 0 else 1
    
    v_loss_PE = torch.sum(loss * PE_mask) / N_PE
    v_loss_SS = torch.sum(loss_bce * SS_mask) / N_SS
    v_loss_CLS = torch.sum(loss_bce * CLS_mask) / N_CLS

    loss_PE = torch.sum(loss * PE_mask) * N_PE / (B**2)
    loss_SS = torch.sum(loss_bce * SS_mask) * N_SS / (B**2)
    loss_CLS = torch.sum(loss_bce * CLS_mask) * N_CLS / (B**2)


    loss = {
        "loss_PE": loss_PE,
        "loss_SS": loss_SS,
        "loss_CLS": loss_CLS,
    }

    # record
    v_loss = {
        "loss_PE": v_loss_PE,
        "loss_SS": v_loss_SS,
        "loss_CLS": v_loss_CLS,
    }

    return loss, v_loss
    
def compute_loss(model, train_data, config):
    '''
    Compute episodic training loss for UniAP.
    [train_data]
        X    : input image of shape (B, T, N, 3, H, W)
        Y    : output label of shape (B, T, N, 1, H, W)
        M    : output mask of shape (B, T, N, 1, H, W)
        t_idx: task index of shape (B, T)
    ''' 

    # 0: PE
    # 1：SS
    # 2: CLS
    X, Y, M, t_idx, _, = train_data

    # split the batches into support and query
    X_S, X_Q = X.split(math.ceil(X.size(2) / 2), dim=2)
    Y_S, Y_Q = Y.split(math.ceil(Y.size(2) / 2), dim=2)
    M_S, M_Q = M.split(math.ceil(M.size(2) / 2), dim=2)

    # ignore masked region in support label
    Y_S_in = torch.where(M_S.bool(), Y_S, torch.ones_like(Y_S) * config.mask_value)

    # compute loss for query images
    Y_Q_pred = model(X_S, Y_S_in, X_Q, t_idx=t_idx, sigmoid=False)
    loss, v_loss = adaptive_loss(Y_Q_pred, Y_Q, M_Q, t_idx)

    
    return loss, v_loss


def normalize_tensor(input_tensor, dim):
    '''
    Normalize Euclidean vector.
    '''
    norm = torch.norm(input_tensor, p='fro', dim=dim, keepdim=True)
    zero_mask = (norm == 0)
    norm[zero_mask] = 1
    out = input_tensor.div(norm)
    out[zero_mask.expand_as(out)] = 0
    return out


def compute_metric(Y, Y_pred, M, task):
    '''
    Compute evaluation metric for each task.
    '''
    # Mean Angle Error
    if task == 'normal':
        pred = normalize_tensor(Y_pred, dim=1)
        gt = normalize_tensor(Y, dim=1)
        deg_diff = torch.rad2deg(2 * torch.atan2(torch.norm(pred - gt, dim=1), torch.norm(pred + gt, dim=1)))
        metric = (M[:, 0] * deg_diff).mean()
        
    elif 'mask' in task :

        acc, miou = measure_pa_miou(num_class=2, gt_image=Y.cpu().int().numpy(), pre_image=Y_pred.cpu().int().numpy())
        print(f"{task}----acc:", acc, "mIOU:", miou)
        metric = acc
    
    elif 'cls' in task:

        cls_pred = reduce(Y_pred.cpu(), 'B C H W -> B 1', 'mean', B =Y_pred.size(0)) 

        cls_gt = reduce(Y.cpu().float(), 'B C H W -> B 1', 'mean', B =Y_pred.size(0))
       
        cls_pred = (cls_pred > 0.5).float()
        
        metric = (cls_pred == cls_gt).float().mean()
   

    # Mean Squared Error
    else:
        metric = (M * F.mse_loss(Y, Y_pred, reduction='none').pow(0.5)).mean()
        
    return metric