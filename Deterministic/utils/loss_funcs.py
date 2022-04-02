#!/usr/bin/env python
# coding: utf-8

import torch
from utils import data_utils



def mpjpe_error(batch_pred,batch_gt): 
 
    batch_pred=batch_pred.contiguous().view(-1,3)
    batch_gt=batch_gt.contiguous().view(-1,3)

    return torch.mean(torch.norm(batch_gt-batch_pred,2,1))
    
    
def final_mpjpe_error(batch_pred,batch_gt): 
    
    batch_pred=batch_pred[:, -1, :, :].contiguous().view(-1,3)
    batch_gt=batch_gt[:, -1, :, :].contiguous().view(-1,3)

    return torch.mean(torch.norm(batch_gt-batch_pred,2,1))




