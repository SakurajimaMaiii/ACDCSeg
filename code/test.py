import numpy as np
import os
import SimpleITK as sitk
import torch


from unet2d import UNet
from utils import test_single_case_dc,test_single_volume_slicebyslice,test_single_case_hd,create_if_not
from preprocess import copy_geometry









if __name__ == "__main__":
    best_model_path = '../log/ACDC0404/model/best.pth'
    data_dir = '../outputs_ACDC'
    gpu = '1'
    test_data_path = data_dir+'/volume'
    NUM_CLS = 4
    save_visualization = True
    results_dir = './results'
    create_if_not(results_dir)
    
    
    
    
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    model = UNet(in_chns=1, class_num=NUM_CLS)
    model.cuda()
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    with open(data_dir+'/test.txt','r') as f:
        test_list = f.readlines()
    test_list = [data_dir+'/volume/'+x.replace('\n','') + '.nii.gz' for x in test_list]
    dice_all = []
    hd_all = []
    for idx,test_img in enumerate(test_list):
        img_path = test_img
        seg_path = img_path.replace('.nii.gz','_gt.nii.gz')
        itkimg = sitk.ReadImage(img_path)
        itkseg = sitk.ReadImage(seg_path)
        spacing = itkimg.GetSpacing()[::-1]
        img = sitk.GetArrayFromImage(itkimg)
        seg = sitk.GetArrayFromImage(itkseg)
        img = (img - np.mean(img)) / np.std(img)
        pred = test_single_volume_slicebyslice(img,model)
        itkpred = sitk.GetImageFromArray(pred)
        itkpred = copy_geometry(itkpred,itkimg)
        dice_arr = test_single_case_dc(pred,seg,NUM_CLS)
        hd_arr = test_single_case_hd(pred,seg,NUM_CLS,spacing)
        #RV_dice,Myo_dice,LV_dice = dice_arr[0],dice_arr[1],dice_arr[2]
        dice_all.append(dice_arr)
        hd_all.append(hd_arr)
        if save_visualization:
            sitk.WriteImage(itkimg,results_dir+'/image_{}.nii.gz'.format(idx+1))
            sitk.WriteImage(itkseg,results_dir+'/label_{}.nii.gz'.format(idx+1))
            sitk.WriteImage(itkpred,results_dir+'/predict_{}.nii.gz'.format(idx+1))
        
        with open(results_dir+'/results.txt','a') as f:
            f.writelines('sample {} RV/Myo/LV/mean dice {}, {}, {}, {}, HD:{},{},{},{}.\n'.format(idx+1,
            dice_arr[0],dice_arr[1],dice_arr[2],(dice_arr[0]+dice_arr[1]+dice_arr[2])/3,
            hd_arr[0],hd_arr[1],hd_arr[2],(hd_arr[0]+hd_arr[1]+hd_arr[2])/3))
            
            
    dice_all = np.array(dice_all)
    rv_dice,myo_dice,lv_dice = np.mean(dice_all,0)
    dice_mean = np.mean(dice_all)
    
    hd_all = np.array(hd_all)
    rv_hd,myo_hd,lv_hd = np.mean(hd_all,0)
    hd_mean = np.mean(hd_all)
    
    print('LV dice: {}, Myo dice: {}, RV dice: {}, mean dice:{}\n'.format(lv_dice,myo_dice,rv_dice,dice_mean))
    print('LV hd: {}, Myo hd: {}, RV hd: {}, mean hd:{}\n'.format(lv_hd,myo_hd,rv_hd,hd_mean))
    with open(results_dir+'/results.txt','a') as f:
        f.writelines('LV dice: {}, Myo dice: {}, RV dice: {}, mean dice:{}\n'.format(lv_dice,myo_dice,rv_dice,dice_mean))
        f.writelines('LV hd: {}, Myo hd: {}, RV hd: {}, mean hd:{}\n'.format(lv_hd,myo_hd,rv_hd,hd_mean))
    
    