# MS-MOS: A Multi-Strip Model for Motion Segmentation

## 📖How to use
### 📦pretrained model
Our pretrained model (validation with the IoU of **_77.6%_**) can be downloaded from [OneDrive](https://1drv.ms/f/c/9eeb7052c0f4734f/Ek9z9MBScOsggJ6y-QAAAAAB__QqT-TAtKKA-2luSsONjQ)
### 📚Dataset 
Download SemanticKITTI dataset from [SemanticKITTI](http://www.semantic-kitti.org/dataset.html#download) (including **Velodyne point clouds**, **calibration data** , **pose data** and **label data**).
#### Preprocessing
After downloading the dataset, the residual maps as the input of the model during training need to be generated.
Run [auto_gen_residual_images.py](./MSMOS-v2/utils/auto_gen_residual_images.py) or [auto_gen_residual_images_mp.py](./MSMOS-v2/utils/auto_gen_residual_images_mp.py)(with multiprocess)
and check that the path is correct before running.

The structure of one of the folders in the entire dataset is as follows:
```
DATAROOT
└── sequences
    ├── 00
    │   ├── poses.txt
    │   ├── calib.txt
    │   ├── times.txt
    │   ├── labels
    │   ├── residual_images_1
    │   ├── residual_images_2
    │   ├── residual_images_3
    │   ├── residual_images_4
    │   ├── residual_images_5
    │   ├── residual_images_6
    │   ├── residual_images_7
    │   ├── residual_images_8
    │   └── velodyne
   ...
```
### 💾Environment
`Linux:`
Ubuntu 18.04, CUDA 11.1+Pytorch 1.7.1

Use conda to create the conda environment and activate it:
```shell
conda env create -f environment.yml
conda activate msmos
```

### 📝Validation and Evaluation
Check the path in [valid.sh](./script/valid.sh) and [evaluate.sh](./script/evaluate.sh).

Then, run them to get the predicted results and IoU in the paper separately:
```shell
bash script/valid.sh
# evaluation after validation
bash script/evaluate.sh
```
You can also use our pre-trained model which has been provided above to validate its performance.

### 👀Visualization
#### Single-frame visualization
Check the path in [visualize.sh](./script/visualize.sh), and run it to visualize the results in 2D and 3D:
```shell
bash script/visualize.sh
```
If -p is empty: only ground truth will be visualized.

If -p set the path of predictions: both ground truth and predictions will be visualized.
#### Get the sequences video
Check the path in [viz_seqVideo.py](./utils/viz_seqVideo.py), and run it to visualize the entire sequence in the form of a video.
