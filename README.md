# ACDCSeg
Medical image segmentation for Automated Cardiac Diagnosis Challenge(ACDC) dataset.This code is clean and easy to follow.
This code is for my own practice, but I also hope it's useful for you.
## Dependcies
We use PyTorch 1.8 and Python 3.8, please follow officical guide to install PyTorch.
Other packages include:
```
simpleitk==2.0.2
numpy==1.19.2
tensorboardX==2.4
medpy==0.4.0
```
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
We use [UNet](https://arxiv.org/abs/1505.04597) as our segmentation backbone.
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
Specify something in `test.py`(line20-27),and run
`python test.py`.
Results will be shown in `code/results`. If you set `save_visualization = True`, visualization results (nii.gz format) will be shown in results too.
We use two metrics for evaluation: Dice and Hausdorff distance. The results are reported as follows:

| | Dice(%) | HD(mm) |
| :------ | :------ | :------ |
| RV | 89.3 | 25.46 |
| Myo | 88.1 | 19.58 |
| LV |94.6 |19.63|
| mean |90.6|21.56|

We don't finetune hyperparameters carefully, so you can easily repreduce this results.
If you use ACDC dataset in your paper, remember to cite paper
```
@article{Bernard2018,
  doi = {10.1109/tmi.2018.2837502},
  url = {https://doi.org/10.1109/tmi.2018.2837502},
  year = {2018},
  month = nov,
  publisher = {Institute of Electrical and Electronics Engineers ({IEEE})},
  volume = {37},
  number = {11},
  pages = {2514--2525},
  author = {Olivier Bernard and Alain Lalande and Clement Zotti and Frederick Cervenansky and Xin Yang and Pheng-Ann Heng and Irem Cetin and Karim Lekadir and Oscar Camara and Miguel Angel Gonzalez Ballester and Gerard Sanroma and Sandy Napel and Steffen Petersen and Georgios Tziritas and Elias Grinias and Mahendra Khened and Varghese Alex Kollerathu and Ganapathy Krishnamurthi and Marc-Michel Rohe and Xavier Pennec and Maxime Sermesant and Fabian Isensee and Paul Jager and Klaus H. Maier-Hein and Peter M. Full and Ivo Wolf and Sandy Engelhardt and Christian F. Baumgartner and Lisa M. Koch and Jelmer M. Wolterink and Ivana Isgum and Yeonggul Jang and Yoonmi Hong and Jay Patravali and Shubham Jain and Olivier Humbert and Pierre-Marc Jodoin},
  title = {Deep Learning Techniques for Automatic {MRI} Cardiac Multi-Structures Segmentation and Diagnosis: Is the Problem Solved?},
  journal = {{IEEE} Transactions on Medical Imaging}
}
```
