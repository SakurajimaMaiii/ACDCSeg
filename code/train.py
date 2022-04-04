"""train"""
#python include
import os
import sys
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import glob
import SimpleITK as sitk
import medpy
import medpy.metric.binary as mmb
#pytorch include
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
#yours
from unet2d import UNet
from dataloaders import ACDCTrainDataset
from utils import create_if_not,set_random,test_single_case_dc,test_single_volume_slicebyslice
from loss import DiceCeLoss


def get_args():
    parser = argparse.ArgumentParser() #can also add description
    parser.add_argument('--gpu', type=str,  default='1', help='GPU to use')
    parser.add_argument('--seed', type=int,  default=0, help='random seed')
    parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    #parser.add_argument('--max_iterations', type=int,  default=6000, help='maximum epoch number to train')
    parser.add_argument('--max_epoch', type=int,  default=200, help='maximum epoch number to train')
    parser.add_argument('--log_dir',type=str,default='../log/0331',help='log dir')
    #parser.add_argument('--exp',type=str,default='baseline',help='experiment name')
    parser.add_argument('--num_class',type=int,default=4,help='numer of class')
    parser.add_argument('--data_dir',type=str,default='../outputs_ACDC',help='dataset path')
    parser.add_argument('--lr',type=float,default=0.001,help='learning rate')
    #parser.add_argument('--crop',type=int,default=1,help='whether crop images')
    parser.add_argument('--flip',type=int,default=1,help='flip or not')
    parser.add_argument('--rot',type=int,default=1,help='rot90 or not')
    #add more
    args = parser.parse_args()
    return args



def main(args):
    print('******make logger******')    
    snapshot_path = args.log_dir
    create_if_not(snapshot_path)
    save_model_path = snapshot_path + '/model'
    create_if_not(save_model_path)
    
    logging.basicConfig(filename=args.log_dir+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    
    
    #model
    print('****** prepare model ******')
    #for prostate
    model = UNet(in_chns=1, class_num=args.num_class)
    model.cuda()
    model.train()
    print('****** dataloader ******')
    #dataloader
    train_dataset  = ACDCTrainDataset(args.data_dir,args.flip,args.rot)
    train_loader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True)
    val_path = args.data_dir+'/volume'
    with open(args.data_dir+'/val.txt','r') as f:
        val_list = f.readlines()
    val_list = [args.data_dir+'/volume/'+x.replace('\n','') + '.nii.gz' for x in val_list]
    print('Validation set has {} volumes.'.format(len(val_list)))
    #optimizer
    print('****** optimizer ******')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    writer = SummaryWriter(snapshot_path+'/log')
    iter_num = 0
    dice_ce_loss = DiceCeLoss(args.num_class)
    best_dice = 0
    best_epoch = 0
    print('******start training******')
    for epoch in range(args.max_epoch):
        #curr_lr = args.lr * (1-epoch/args.max_epoch)**0.9
        #update_lr(optimizer,curr_lr)
        for idx, sampled_batch in enumerate(train_loader):
            image,label = sampled_batch
            image,label = image.cuda(),label.cuda()
            outputs = model(image)
            loss = dice_ce_loss(outputs,label)
             
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #record loss
            iter_num = iter_num + 1
            writer.add_scalar('loss/loss', loss, iter_num)
            
            logging.info('iteration %d : loss : %f'% (iter_num, loss.item()))

        #val
        model.eval()
        dice_all = []
        for val_img in val_list:
            img_path = val_img
            seg_path = img_path.replace('.nii.gz','_gt.nii.gz')
            img = sitk.ReadImage(img_path)
            seg = sitk.ReadImage(seg_path)
            img = sitk.GetArrayFromImage(img).astype(np.float32)
            seg = sitk.GetArrayFromImage(seg).astype(np.int8)
            img = (img - np.mean(img)) / np.std(img)
            pred = test_single_volume_slicebyslice(img,model)
            dice_arr = test_single_case_dc(pred,seg,args.num_class)
            #RV_dice,Myo_dice,LV_dice = dice_arr[0],dice_arr[1],dice_arr[2]
            dice_all.append(dice_arr)
        dice_all = np.array(dice_all)
        rv_dice,myo_dice,lv_dice = np.mean(dice_all,0)
        dice_mean = np.mean(dice_all)
            
        
        model.train()
        logging.info('Epoch val mean dice:%f'% dice_mean)
        writer.add_scalar('val/dice', dice_mean, epoch)
        writer.add_scalar('val/dice_lv',lv_dice, epoch)
        writer.add_scalar('val/dice_rv',rv_dice, epoch)
        writer.add_scalar('val/dice_myo',myo_dice, epoch)
        if dice_mean>=best_dice:
           best_epoch = epoch
           torch.save(model.state_dict(), save_model_path+'/best.pth')
        
        
        
        torch.save(model.state_dict(), save_model_path+'/epoch_{}.pth'.format(epoch))
        



    writer.close()
    print('******training finished, writer closed******')
            

if __name__ == "__main__":
    args = get_args()
    create_if_not(args.log_dir)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print('******fix random seed %d ******' % args.seed)
    set_random(args.seed) 
    main(args)