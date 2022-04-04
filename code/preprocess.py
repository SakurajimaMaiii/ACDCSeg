# -*- coding: utf-8 -*-

import SimpleITK as sitk
import numpy as np
import os
from skimage.transform import resize
import glob
#resample to 1.5x1.5 slice by slice

def load_nii(path):
    itkimg = sitk.ReadImage(path)
    img = sitk.GetArrayFromImage(itkimg)
    return img

def resize_segmentation(segmentation, new_shape, order=0, cval=0):
    '''
    Resizes a segmentation map. Supports all orders (see skimage documentation). Will transform segmentation map to one
    hot encoding which is resized and transformed back to a segmentation map.
    This prevents interpolation artifacts ([0, 0, 2] -> [0, 1, 2])
    :param segmentation:
    :param new_shape:
    :param order:
    :return:
    '''
    tpe = segmentation.dtype
    unique_labels = np.unique(segmentation)
    assert len(segmentation.shape) == len(new_shape), "new shape must have same dimensionality as segmentation"
    if order == 0:
        return resize(segmentation.astype(float), new_shape, order, mode="constant", cval=cval, clip=True, anti_aliasing=False).astype(tpe)
    else:
        reshaped = np.zeros(new_shape, dtype=segmentation.dtype)

        for i, c in enumerate(unique_labels):
            mask = segmentation == c
            reshaped_multihot = resize(mask.astype(float), new_shape, order, mode="edge", clip=True, anti_aliasing=False)
            reshaped[reshaped_multihot >= 0.5] = c
        return reshaped
 
        
def resample(img,old_spacing,new_spacing=(1.0,1.0,1.0),data_type='img'):
    shape = np.shape(img)
    new_shape = np.round(((np.array(old_spacing) / np.array(new_spacing)).astype(float) * shape)).astype(int)
    if data_type == 'img':
        new_data = resize(img,new_shape,order=3)
    elif data_type == 'seg':
        new_data = resize_segmentation(img,new_shape)
    else:
        raise NotImplementedError('data type is not implemented')
    return new_data

def create_if_not(path):
    #create path if not exist
    if not os.path.exists(path):
        os.makedirs(path)
        
def copy_geometry(image: sitk.Image, ref: sitk.Image):
    image.SetOrigin(ref.GetOrigin())
    image.SetDirection(ref.GetDirection())
    image.SetSpacing(ref.GetSpacing())
    return image

def preprocess_one_case(img_path,seg_path,output_spacing,crop_size):
    #special for this dataset
    itkimg = sitk.ReadImage(img_path)
    itkseg = sitk.ReadImage(seg_path)
    old_spacing = itkimg.GetSpacing()[::-1]
    #print(old_spacing)
    old_spacing_z = old_spacing[0]
    old_spacing = old_spacing[1:]
    npimg = sitk.GetArrayFromImage(itkimg)
    npseg = sitk.GetArrayFromImage(itkseg)
    npimg = npimg.astype(np.float32)
    npseg = npseg.astype(np.int8)
    #print(np.shape(npimg))
    num_slices = np.shape(npimg)[0]
    new_img = []
    new_seg = []
    for i in range(num_slices):
        img2d = npimg[i]
        seg2d = npseg[i]
        new_img2d = resample(img2d,old_spacing,output_spacing)
        new_seg2d = resample(seg2d,old_spacing,output_spacing,'seg')
        new_img2d = center_crop_2d(new_img2d,crop_size)
        new_seg2d = center_crop_2d(new_seg2d,crop_size)
        new_img.append(new_img2d)
        new_seg.append(new_seg2d)
    new_img = np.stack(new_img)
    new_seg = np.stack(new_seg)
    #print(np.shape(new_img))
    new_itkimg = sitk.GetImageFromArray(new_img)
    new_itkseg = sitk.GetImageFromArray(new_seg)
    new_itkimg = copy_geometry(new_itkimg,itkimg)
    new_itkseg = copy_geometry(new_itkseg,itkseg)
    new_itkimg.SetSpacing((output_spacing[1],output_spacing[0],old_spacing_z))
    new_itkseg.SetSpacing((output_spacing[1],output_spacing[0],old_spacing_z))
    return new_itkimg,new_itkseg
    
def center_crop_2d(data,crop_size=(128,128)):
    """
    center crop for 2d data
    input must be[C,H,W] or [H,W](will be expand to [1,H,W])
    output shape will be same as input.
    """
    dims = len(np.shape(data))
    if dims==2:
        data = np.expand_dims(data,axis=0)
    
    c,x,y = np.shape(data)
    #pad if necessary
    if x<=crop_size[0] or y<=crop_size[1]:
        px = max((crop_size[0] - x) // 2 + 3, 0)
        py = max((crop_size[1] - y) // 2 + 3, 0)
        data = np.pad(data,[(0,0),(px,px),(py,py)],mode='constant', constant_values=0)
        print('data size is smaller than crop size, we pad it')
        
    c,x,y = np.shape(data)
    x1 = int(round((x - crop_size[0]) / 2.))
    y1 = int(round((y - crop_size[1]) / 2.))
    
    data = data[:,x1:x1+crop_size[0],y1:y1+crop_size[1]]
    
    if dims==2:
        return data[0]
    return data    
    
    
   
if __name__ == '__main__':
    base_dir = '../training' #specify it if need
    outputs_dir = '../outputs_ACDC'
    #preprocessed dataset would be organized as follows:
    #--'outputs_ACDC'
    #   --volume
    #   --slice(just contain slices for training)
    #   --train_list.txt
    #   --val_list.txt
    #   --test_list.txt
    create_if_not(outputs_dir)
    create_if_not(outputs_dir+'/volume')
    create_if_not(outputs_dir+'/slice')
    target_spacing = (1.5,1.5)
    CROP_SIZE = (192,192)
        
    patient_list = os.listdir(base_dir)
    #print(len(patient_list))
    
    

    for pid in patient_list:
        pid_path = os.path.join(base_dir,pid)
        gt_list = glob.glob(pid_path+'/*gt.nii.gz')
        gt_path1 = gt_list[0]
        gt_path2 = gt_list[1]
        img_path1 = gt_path1.replace('_gt','')
        img_path2 = gt_path2.replace('_gt','')
        #resample slice by slice and crop to 224x224
        new_img1,new_seg1 = preprocess_one_case(img_path1,gt_path1,target_spacing,CROP_SIZE)
        new_img2,new_seg2 = preprocess_one_case(img_path2,gt_path2,target_spacing,CROP_SIZE)
        sitk.WriteImage(new_img1,outputs_dir+'/volume/%s_frame01.nii.gz'%pid)
        sitk.WriteImage(new_seg1,outputs_dir+'/volume/%s_frame01_gt.nii.gz'%pid)
        sitk.WriteImage(new_img2,outputs_dir+'/volume/%s_frame02.nii.gz'%pid)
        sitk.WriteImage(new_seg2,outputs_dir+'/volume/%s_frame02_gt.nii.gz'%pid)
        print('%s saved'%pid)
    #generate slices
    with open(outputs_dir+'/train.txt') as f:
        lines = f.readlines()
      
    num_slices = 0
    for line in lines:
        line = line.replace('\n','')
        img_path = outputs_dir+'/volume/'+line+'.nii.gz'
        seg_path = outputs_dir+'/volume/'+line+'_gt.nii.gz'
        itkimg = sitk.ReadImage(img_path)
        img = sitk.GetArrayFromImage(itkimg)
        itkseg = sitk.ReadImage(seg_path)
        seg = sitk.GetArrayFromImage(itkseg)
        img = (img - np.mean(img)) / np.std(img)
        for z in range(np.shape(img)[0]):
            img2d = img[z]
            seg2d = seg[z]
            data = np.stack([img2d,seg2d])
            num_slices += 1
            np.save(outputs_dir+'/slice/%4d.npy'%num_slices,data)
    print('totaly slices: %d'%num_slices)
        

        
        
        
        
        
        
        
        
    
    
      