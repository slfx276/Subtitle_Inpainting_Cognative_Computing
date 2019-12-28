# Subtitle_Inpainting_Cognative_Computing

## Purpose  
The purpose of this project is to remove the subtitles of input video.  

- First, we need to detect breakpoint of different scenes, because the inpainting scheme would utilize temporal information. We should not confuse them with siginificantly different frames. And [PySceneDetect-0.5.1.1](https://github.com/Breakthrough/PySceneDetect/) is used in this scene detection task.  

- Second, we have to detect the regions of subtitle, and make subtitle masks corresponding to the images in order to offer a reference of inpainting region to inpainting model. We use ["Character Region Awareness for Text Detection"](https://github.com/clovaai/CRAFT-pytorch) for this task.


- Finally, we use Free Form Video Inpainting proposed by [Free-form Video Inpainting with 3D Gated Convolution and Temporal PatchGAN. Chang et al. ICCV 2019.](https://github.com/amjltc295/Free-Form-Video-Inpainting) , which use 3D convolutional filter to consider temporal information near by the specific frame.  

## Environment Requirements  
detailrequirements can be check in website links of these packages.  
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
    
## Usage

initially, your directory in current folder(e.g. MovieSubtitle_Dataset) should be like:  
```
MovieSubtitle_Dataset
├── VideoCapture.py
├── PySceneDetect-0.5.1.1
└── CRAFT-pytorch-master
```

after pull in the movie files you want to test, the directory should be like.  

```
MovieSubtitle_Dataset
├── VideoCapture.py
├── PySceneDetect-0.5.1.1
├── CRAFT-pytorch-master
├── movie1
│   ├── video_file_of_movie1.avi (or .mp4, .mkv)
│   └── subtitle_file_of_movie1.srt
└── movie2
    └── video_file_of_movie2.avi (or .mp4, .mkv)
```

- **Create captures and regerence masks for inpainting inference**
```
python VideoCapture.py -cls True -is 256 256
```
- **Create training data for Free Form Inpainting (then movies should have .srt files ! )**
```
python VideoCapture.py -i False -cls True -is 256 256
```
then the results captures and masks would be saved in folders videos and folders masks as :  
```
MovieSubtitle_Dataset
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
usage: Create inputs of Free Form Inpainting Game. [-h] [-i INFERENCE]
                                                   [-is IMG_SIZE [IMG_SIZE ...]]
                                                   [-sn SAMPLE_NUM]
                                                   [-gap FRAME_GAP]
                                                   [-sc SCENE_DETECTION]
                                                   [-fs FONT_SIZE]
                                                   [-fc FONT_COLOR]
                                                   [-cls CLEAN_OLD_FILES]

optional arguments:
  -h, --help            show this help message and exit
  -i INFERENCE, --inference INFERENCE
                        created data is for inference or training Free Form
                        Inpainting GAN.
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
  -cls CLEAN_OLD_FILES, --cleanfile CLEAN_OLD_FILES
                        clean old data before creating new ones
```




## Reference
- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect/)
- [PySceneDetect](https://pyscenedetect.readthedocs.io/en/latest/changelog/)
- ["Character Region Awareness for Text Detection"](https://github.com/clovaai/CRAFT-pytorch)
- [Free-form Video Inpainting with 3D Gated Convolution and Temporal PatchGAN. Chang et al. ICCV 2019.](https://github.com/amjltc295/Free-Form-Video-Inpainting)
