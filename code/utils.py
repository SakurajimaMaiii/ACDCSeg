import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random
import copy
import os
import medpy.metric.binary as mmb



def create_if_not(path):
    #create path if not exist
    if not os.path.exists(path):
        os.makedirs(path)



def set_random(seed_id=1234):
    #set random seed for reproduce
    random.seed(seed_id)
    np.random.seed(seed_id)
    torch.manual_seed(seed_id)   #for cpu
    torch.cuda.manual_seed_all(seed_id) #for GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    
def test_single_case_dc(pred,label,num_class):
    #
    dice_array = []
    for i in range(1,num_class):
        predi = pred==i
        labeli = label==i
        dice_i = mmb.dc(predi,labeli)
        dice_array.append(dice_i)
    return dice_array
    
    
def test_single_case_hd(pred,label,num_class,spacing):
    #
    dice_array = []
    for i in range(1,num_class):
        predi = pred==i
        labeli = label==i
        dice_i = mmb.hd(predi,labeli,voxelspacing=spacing)
        dice_array.append(dice_i)
    return dice_array
        
def test_single_volume_slicebyslice(img,model):
    model.cuda()
    model.eval()
    
    #img must be  zyx
    num_slices = np.shape(img)[0]
    prediction_3d = []
    for i in range(num_slices):
        img2d = img[i]
        img2d = torch.from_numpy(img2d).unsqueeze(0).unsqueeze(0).cuda()
        with torch.no_grad():
            pred_2d = model(img2d) 
            pred_2d = torch.argmax(pred_2d,1)
            pred_2d = pred_2d.detach().cpu().numpy()[0]
        prediction_3d.append(pred_2d)  
    prediction_3d = np.stack(prediction_3d)
    return prediction_3d
