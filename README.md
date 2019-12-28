# Subtitle_Inpainting_Cognative_Computing

## Purpose  
The purpose of this project is to remove the subtitles of input video.  

- First, we need to detect breakpoint of different scenes, because the inpainting scheme would utilize temporal information. We should not confuse them with siginificantly different frames. And [PySceneDetect-0.5.1.1](https://github.com/Breakthrough/PySceneDetect/) is used in this scene detection task.  

- Second, we have to detect the regions of subtitle, and make subtitle masks corresponding to the images in order to offer a reference of inpainting region to inpainting model. We use ["Character Region Awareness for Text Detection"](https://github.com/clovaai/CRAFT-pytorch) for this task.


- Finally, we use Free Form Video Inpainting proposed by [Free-form Video Inpainting with 3D Gated Convolution and Temporal PatchGAN. Chang et al. ICCV 2019.](https://github.com/amjltc295/Free-Form-Video-Inpainting) , which use 3D convolutional filter to consider temporal information near by the specific frame.  

## Usage

## Reference
- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect/)
- [PySceneDetect](https://pyscenedetect.readthedocs.io/en/latest/changelog/)
- ["Character Region Awareness for Text Detection"](https://github.com/clovaai/CRAFT-pytorch)
- [Free-form Video Inpainting with 3D Gated Convolution and Temporal PatchGAN. Chang et al. ICCV 2019.](https://github.com/amjltc295/Free-Form-Video-Inpainting)
