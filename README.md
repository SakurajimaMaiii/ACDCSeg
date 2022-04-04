# ACDCSeg
Medical image segmentation for Automated Cardiac Diagnosis Challenge(ACDC) dataset.This code is clean and easy to follow.
This code is for my own practice, but I also hope it's useful for you.
## Dependcies
We use PyTorch 1.8, please follow officical guide to install PyTorch.
Other packages include SimpleITK,numpy,tensorboardX,medpy.
## Prepare dataset
Please download dataset from [ACDC challenge](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html) and organize data as follows
```
--code
--training #this file includes all data
  -- patient001
    --Info.cfg
    --patient001_4d.nii.gz
    --patient001_frame01.nii.gz
    ------
```
then 
```
cd code
run preprocess.py
```
preprocessed dataset will appear in outputs_ACDC.Finally,you will see
```
--code
--training #this file includes all data
--outputs_ACDC
```
## Train
You can directly run train.py, for example:
```
python train.py --log_dir ../log/ACDC0404
```
Log file will be in log/ACDC0404.
Log file would be organized as follows:
```
--code
--log
  --ACDC0404
    --log #include tensorboard file
    --model #include model
    --log.txt    
--training #this file includes all data
--outputs_ACDC
```
## Test
Specify something in `test.py'(line20-27),and run
'python test.py'.
Results will be shown in code/results. If you set 'save_visualization = True', visualization results (nii.gz format) will be shown in results too.
We use two metrics for evaluation: Dice and Hausdorff distance.The results are reported as follows:

| | Dice(%) | HD(mm) |
| :------ | :------ | :------ |
| RV | 89.3 | 25.46 |
| Myo | 88.1 | 19.58 |
| LV |94.6 |19.63|
| mean |90.6|21.56|

We don't finetune hyperparameters, so you can easily repreduce this results.
If you use ACDC dataset in your paper, remember to cite paper [Deep Learning Techniques for Automatic MRI Cardiac Multi-Structures Segmentation and Diagnosis: Is the Problem Solved?](https://ieeexplore.ieee.org/document/8360453).
