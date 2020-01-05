# Subtitle_Inpainting_Cognative_Computing

## Purpose  
The purpose of this project is to remove the subtitles of input video.  

- First, we need to detect breakpoint of different scenes, because the inpainting scheme would utilize temporal information. We should not confuse them with significantly different frames. And [PySceneDetect-0.5.1.1](https://github.com/Breakthrough/PySceneDetect/) is used in this scene detection task.  

- Second, we have to detect the regions of subtitle, and make subtitle masks corresponding to the images in order to offer a reference of inpainting region to inpainting model. We use ["Character Region Awareness for Text Detection"](https://github.com/clovaai/CRAFT-pytorch) for this task.


- Finally, we use Free Form Video Inpainting proposed by [Free-form Video Inpainting with 3D Gated Convolution and Temporal PatchGAN. Chang et al. ICCV 2019.](https://github.com/amjltc295/Free-Form-Video-Inpainting) , which use 3D convolutional filter to consider temporal information near by the specific frame.  

## Download
Video preprocessing testing folder includes **1. PySceneDetec and 2. Text Detection(CRAFT)** packages can be downloaded [Here](https://drive.google.com/open?id=1ln3P9sjMEAL9o6Aanm46poLW6qPID8yO)  
  
For 3. Free Form Inpainting GAN, you could check their github repository.

## Environment Requirements  
detail requirements can be check in website links of these packages.  
1. PySceneDetect
    - OpenCV
    - Numpy
    - Click
    - tqdm

2.  CRAFT-pytorch
    - Pytorch >= 0.4.1
    - torchvision >= 0.2.1
    - opencv-python >= 3.4.2
    - scikit-image==0.14.2
    - scipy==1.1.0
3.  Free Form Inpainting
    - check its requirements.txt  
    
**all my packages in virtual environment is in environment.yml**  
```
$ conda env create -n videoPreprocessing -f environment.yml  
$ conda activate videoPreprocessing
```


## Notification  
- This code is executed in Win10, it may occur some errors in other OSs. Especially I use windows internal font type in function Make_masks( ) QQ . And if languags of .srt files is not English , I don't really know whether it would cause some errors like *UnicodeDecodeError*.  
- If the resizing image size is too small, it would affect text detection result.  
- If your video have **no internal subtitle** instead of only a .srt file, than using inference command is wasting time.  
Because it won't create any meaningful mask of subtitles, instead of white image and some masks of frames which have text content(not subtitles).  
- Movie with only black and white color seems not good for our scene detection method.  

## Usage

initially, your directory in current folder(e.g. temp) should be like:  
```
temp
├── VideoCapture.py
├── PySceneDetect-0.5.1.1
└── CRAFT-pytorch-master
```

after pull in the movie files you want to test, the directory should be like.  

```
temp
├── VideoCapture.py
├── PySceneDetect-0.5.1.1
├── CRAFT-pytorch-master
├── movie1
│   ├── video_file_of_movie1.avi (or .mp4, .mkv)
│   └── subtitle_file_of_movie1.srt
└── movie2
    └── video_file_of_movie2.avi (or .mp4, .mkv)
```

- **Create captures and reference masks for inpainting inference**
```
$ python VideoCapture.py --inference --cleanfile -is 256 256
```
- **Create training data for Free Form Inpainting (then movies should have .srt files ! )**
```
$ python VideoCapture.py --cleanfile -is 256 256
```
( Note that if the resizing image size is too small, it would affect text detection result. )  
then the results captures and masks would be saved in folder videos and folder masks as :  
```
temp
├── VideoCapture.py
├── PySceneDetect-0.5.1.1
├── CRAFT-pytorch-master
├── movie1
├── movie2
└── MovieDataset
    └── Test
        ├── videos
        └── masks
```
mask images and capture images would have similar image names.  
The remaining work includes feeding these created new inputs into Free Form Inpainting GAN, analysing the experiment results and study the methods of scenes detection and text detection.  


**Arguments for VideoCapture.py**
```
usage: Create inputs of Free Form Inpainting Game. [-h] [-i] [-cls]
                                                   [-is IMG_SIZE [IMG_SIZE ...]]
                                                   [-sn SAMPLE_NUM]
                                                   [-gap FRAME_GAP]
                                                   [-sc SCENE_DETECTION]
                                                   [-fs FONT_SIZE]
                                                   [-fc FONT_COLOR]

optional arguments:
  -h, --help            show this help message and exit
  -i, --inference       created data is for inference or training Free Form
                        Inpainting GAN.
  -cls, --cleanfile     clean old data before creating new ones
  -is IMG_SIZE [IMG_SIZE ...], --imgsize IMG_SIZE [IMG_SIZE ...]
                        image size that you want to resize the video frame
                        size to.
  -sn SAMPLE_NUM, --samplenumber SAMPLE_NUM
                        you can determine the samples length for training Free
                        Form Inpainting GAN.
  -gap FRAME_GAP, --framegap FRAME_GAP
                        frame gap between two video frames that we want to
                        capture
  -sc SCENE_DETECTION, --scene SCENE_DETECTION
                        do you want to detect different scenes before capture
                        the video frames
  -fs FONT_SIZE, --fontsize FONT_SIZE
                        font size in mask images that you created for training
                        Free Form Inpainting GAN.
  -fc FONT_COLOR, --fontcolor FONT_COLOR
                        font color in mask images that you created for
                        training Free Form Inpainting GAN.
```




## Reference
- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect/)
- [PySceneDetect](https://pyscenedetect.readthedocs.io/en/latest/changelog/)
- ["Character Region Awareness for Text Detection"](https://github.com/clovaai/CRAFT-pytorch)
- [Free-form Video Inpainting with 3D Gated Convolution and Temporal PatchGAN. Chang et al. ICCV 2019.](https://github.com/amjltc295/Free-Form-Video-Inpainting)
