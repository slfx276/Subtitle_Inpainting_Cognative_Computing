import os 
import logging
import sys
import cv2
import re
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from tqdm import tqdm
import time as t
import random
import string
import numpy as np
import argparse
import shutil


finished_movie_count = 0
mask_folder_count = 0

def get_parser():
    """
    Creates a new argument parser.
    """
    parser = argparse.ArgumentParser('Create inputs of Free Form Inpainting Game.')
    parser.add_argument("-i", "--inference", type=bool, default=True, dest="inference",
                            help="created data is for inference or training Free Form Inpainting GAN.")
    parser.add_argument("-is", "--imgsize", type=int, default=(128,128), dest="img_size", nargs='+', 
                            help="image size that you want to resize the video frame size to.")
    parser.add_argument("-sn", "--samplenumber", type=int, default=16, dest="sample_num",
                            help="you can determine the samples length for training Free Form Inpainting GAN.")
    parser.add_argument("-gap", "--framegap", type=int, default=5, dest="frame_gap", 
                            help="frame gap between two video frames that we want to capture")
    parser.add_argument("-sc", "--scene", type=bool, default=True, dest="scene_detection", 
                            help="do you want to detect different scenes before capture the video frames")
    parser.add_argument("-fs", "--fontsize", type=int, default=7, dest="font_size", 
                            help="font size in mask images that you created for training Free Form Inpainting GAN.")
    parser.add_argument("-fc", "--fontcolor", type=str, default="black", dest="font_color", 
                            help="font color in mask images that you created for training Free Form Inpainting GAN.")
    parser.add_argument("-cls", "--cleanfile", type=bool, default=False, dest="clean_old_files", 
                            help="clean old data before creating new ones")

    return parser.parse_args()

def randomString(stringLength=10):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))

def Create_Logger(log_level = logging.DEBUG):

    # Create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    # Create STDERR handler
    handler = logging.StreamHandler(sys.stderr)
    # ch.setLevel(logging.DEBUG)

    # Create formatter and add it to the handler
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # Set STDERR handler as the only handler 
    logger.handlers = [handler]
    return logger



def Check_SRT_File(Inference = False):
    '''
        Check how many movie folders need to be processed in current directory,
        and turn their .srt subtitle file into .txt format (if there is .srt).

        Return
            subtitle_file_path_dict: dict,
                keys are movie names (folder name), values are paths to movie's subtitle txt file.
            movie_path_list: list,
                return list of movies which have subtitle .srt file, if a movie has no subtitle,
                it is not included in.


    '''

    exclude_list = [".ipynb_checkpoints", "FaceForensics", "FVI", "VideoCapture.ipynb","CRAFT-pytorch-master",
                       "VideoCapture.py", "PySceneDetect-0.5.1.1", 'MovieDataset']
    movie_list = os.listdir()
    for item in exclude_list:
        if item in movie_list:
            movie_list.remove(item)
    for item in movie_list:
        if not os.path.isdir(item):
            movie_list.remove(item)


    movie_count = 0
    subtitle_file_path_dict = dict()
    movie_path_list = list()
    print(movie_list)
    
    if Inference:
        logger.info(f"Inference movie list: {movie_list}")
        return subtitle_file_path_dict, movie_list

    for movie in movie_list:
        movie_files = os.listdir(movie)
        # 檢查檔案副檔名
        for file in movie_files:
            if file.split(".")[-1] == "srt":      
                
                # transform .srt to .txt
                subtitle = list()
                try:
                    with open(os.path.join(movie, file), "r", encoding="utf-8") as f:
                        subtitle = f.readlines()

                    with open(os.path.join(movie, movie + ".txt"), "w", encoding="utf-8") as f:
                        f.writelines(subtitle)
                        
                    subtitle_file_path_dict[movie] = (os.path.join(movie, movie + ".txt"))
                    movie_path_list.append(movie)
                    logger.debug(movie)

                    
                # For some other binary files
                except:
                    with open(os.path.join(movie, file), "rb") as f:
                        subtitle = f.readlines()

                    with open(os.path.join(movie, movie + ".txt"), "wb") as f:
                        f.writelines(subtitle)
                                              
                    subtitle_file_path_dict[movie] = (os.path.join(movie, movie + ".txt"))
                    movie_path_list.append(movie)
                    logger.debug("Binary" + movie)
                    
                finally:
                    movie_count += 1
        
                break
                                              
    logger.debug(f"Already create subtitle file of {movie_count} movies, sub_movie_list: {movie_path_list}")
    
    # movie list which have .srt subtitle file
    return subtitle_file_path_dict, movie_path_list

def Make_masks(mask_save_path, text_content, img_size=(128,128), show = False, color='black', font_size=7, frame_has_no_sub=False):
    # Write Name on the Image
    img = np.zeros(img_size, np.uint8)
    img.fill(255)
    img = Image.fromarray(img)

    if frame_has_no_sub:
        img.save(mask_save_path)

    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("C:/WINDOWS/Fonts/Arial.ttf", font_size)
    
    # change line
    if len(text_content) > 30:
        string = text_content.split(' ')
        line1 = ' '.join(string[:6])
        line2 = ' '.join(string[6:])
        draw.text((img_size[1]//2-len(line1)*3//2, img_size[0]-30), line1, font=font, fill=color)
        draw.text((img_size[1]//2-len(line2)*3//2, img_size[0]-20), line2, font=font, fill=color)
    else:
        draw.text((img_size[1]//2-len(text_content)*4//2, img_size[0]-20), text_content, font=font, fill=color)
    
    img.save(mask_save_path)

    
    if show:
        img.show()

    return img

def PyScene_detect(movie, video_file):
    '''
    Use github repository (PySceneDetect) to detect the breakpoint frame of different scenes. 

    Argument
        movie: str
            movie name (folder name in MovieSubtitle_Dataset).
        video_file: str
            the video file name with extension .avi, .mp4 and .mkv.
    
    Return
        the frame number list of break point frame
    '''

    import pandas as pd
    
    # run PySceneDetect
    os.chdir("PySceneDetect-0.5.1.1")
    os.system(f'scenedetect --input {os.path.join("../", movie, video_file)} detect-content list-scenes')
    # read in the dectection result csv
    df = pd.read_csv(video_file[:-4] + "-Scenes.csv")
    break_point_list = list(df.iloc[1:,4].astype("int32"))
    os.chdir("..")

    logger.info(f"SceneDetection Break Point frame {break_point_list}")
    return break_point_list
            
def Detect_Text(created_folder_list, show_mask=False):
    '''
    Detect text bounding box in the frame captures folder,
    and save the correspoinding mask images to path of masks.

    Argument
        created_folder_list: 
            while inference, each scene of movie would create a folder,
            so pass in the folder_name list. (just folder name, not whole path)
        
        show_mask: Bool,
             you can show the mask image immediately.
    '''

    for folder_name in created_folder_list:
        mask_save_path = os.path.join("MovieDataset", "Test", "masks", folder_name)
        if not os.path.exists(mask_save_path):
                os.makedirs(mask_save_path)
        # path to Captured Frames of Movie
        test_folder_path = os.path.join("MovieDataset", "Test", "videos", folder_name)
        img_name_list = os.listdir(test_folder_path)


        os.chdir("CRAFT-pytorch-master")
        os.system(f"python test.py --trained_model=craft_mlt_25k.pth --test_folder=../{test_folder_path}")
        os.chdir("result")
        logger.debug(f"img_name_list: {img_name_list}")

        # make mask of each image
        for img_name in img_name_list:
            name = "res_" + img_name[:-4]
            with open(name + ".txt", "r") as f:
                result = f.readlines()

            # filtering file content
            result_list = list()
            for idx in range(len(result)):
                if result[idx] == "\n":
                    continue
                else:
                    result_list.append(result[idx].strip("\n").split(","))

            # create bounding box coordinate of points
            text_region = list()
            for idx in range(len(result_list)):
                bounding_box_list = list()
                # extract 4 points of retangle
                x = list()
                y = list()
                for i in range(4):
                    x.append(int(result_list[idx][2*i]))
                    y.append(int(result_list[idx][2*i + 1]))

                bounding_box_list.append([ (min(x), min(y)), (max(x), max(y)) ])
                text_region.append(bounding_box_list)
                logger.debug(f"Bounding Box of {img_name} -> {text_region}")

            # create mask of this image
            img_size = cv2.imread(name + ".jpg").shape
            img = np.zeros(img_size, np.uint8)
            img.fill(255)
            img = Image.fromarray(img)
            draw = ImageDraw.Draw(img)
            for bounding_box in text_region:
                for coordinary in bounding_box:
                        # draw.rectangle(coordinary, fill=0)
                        draw.rectangle(coordinary, fill = 0)
            img.save("../../" + mask_save_path +"/"+ img_name + ".jpg")

            if show_mask:
                img.show()

        os.chdir("../..")




def Extract_Video_info(args, movie, video_file, subtitle_file_path_dict):
    '''
    This function is especially for createing training dataset.
    Detect the subtitle file and calculate the start frame and end frame of a subtitle,
    and then save frame Captures and masks with only subtitle simualtaneously.

    It would go through wholw frames of a video, so it needs some time.

    Argument:
        movie: str,
            movie name (folder name in MovieSubtitle_Dataset).
        video_file: str,
            the video file name with extension .avi, .mp4 and .mkv.
            
    '''
    sample_num = args.sample_num
    frame_gap = args.frame_gap
    image_resize = tuple(args.img_size)
    Scene_detection = args.scene_detection
    
    video_path = os.path.join(movie, video_file)
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    image_size = (int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(video.get(cv2.CAP_PROP_FRAME_WIDTH)))
    logger.info("{} => fps = {}, total_frames = {}, original image_size = {}, resize to {}".format(video_path, round(fps), total_frames, image_size, image_resize))


    # Shot Detection, if necessary.
    if Scene_detection:
        break_point_list = PyScene_detect(movie, video_file)

    time_token_idx = list()
    frameIdx_with_sub = list()
    end_frameIdx_with_sub = list()
    frame_subtitle_dict = dict()

    # read subtitle file of this movie
    with open(subtitle_file_path_dict[movie], "r", encoding="utf-8") as f:
        mapping = f.readlines()

        for idx in range(len(mapping)):
            if len(mapping[idx].split("-->")) == 2:
                time_token_idx.append(idx)
                logger.debug(mapping[idx])

        for idx in tqdm(range(len(time_token_idx))):
            # calculate the start frame and ending frame of subtitle 
            time = mapping[time_token_idx[idx]][:8]
            end_time = mapping[time_token_idx[idx]].split("--> ")[1][:8]
            hour, minute, second = int(time[:2]), int(time[3:5]), int(time[6:])
            end_hour, end_minute, end_second = int(end_time[:2]), int(end_time[3:5]), int(end_time[6:])
            # calculate frame number with subtitle
            frame_number = ((hour * 60 + minute) * 60 + second) * round(fps)
            end_frame_number = ((end_hour * 60 + end_minute) * 60 + end_second) * round(fps)

            frameIdx_with_sub.append(frame_number)
            end_frameIdx_with_sub.append(end_frame_number)
            logger.debug(f"Frame Number-> {frame_number}, {end_frame_number}")

            subtitle = mapping[time_token_idx[idx] + 1].strip("\n \t</i>")
            # if subtitle cost 2 rows
            if idx != len(time_token_idx)-1 and time_token_idx[idx] + 3 != time_token_idx[idx + 1]:
                subtitle = subtitle + " " + mapping[time_token_idx[idx] + 2].strip("\n \t</i>")
            for frame in range(frame_number, end_frame_number+1):
                frame_subtitle_dict[frame] = subtitle

            logger.debug(frame_number, end_frame_number, subtitle)
        
    logger.debug("There are {} frames need to be extracted.".format(len(frameIdx_with_sub)))
    logger.debug(frameIdx_with_sub, frame_subtitle_dict)

    # make Video Capture
    if video.isOpened():
        rval, frame = video.read()
    else:
        rval = False
        print("open video failed")

    # define path format
    count = 1    
    global mask_folder_count
    pbar = tqdm(total = total_frames)
    
    # Create Capture of Video

    t1 = t.time()
    while rval:
        if count in frame_subtitle_dict.keys() and count < total_frames - (frame_gap * sample_num):
            # if the frame samples would cross over different scenes, drop them.
            if Scene_detection:
                drop = False
                for breakpoint_frame in break_point_list:
                    if count < breakpoint_frame < count + frame_gap * sample_num:
                        drop = True
                        break
                if drop == True:
                    rval, frame = video.read()
                    count = count+1
                    pbar.update(1)
                    continue

            # Create new folder of samples
            mask_number = 1
            folder_name = randomString() + "_" + str(finished_movie_count)
            capture_save_path = os.path.join("MovieDataset", "Test", "videos", folder_name)
            mask_save_path = os.path.join("MovieDataset", "Test", "masks", "test_object_like", "object_like_" + str(mask_folder_count) + "_"+ str(mask_folder_count))
            if not os.path.exists(capture_save_path):
                os.makedirs(capture_save_path)
                logger.info(f"{movie} in folder {folder_name}")
            
            if not os.path.exists(mask_save_path):
                os.makedirs(mask_save_path)
            
            mask_folder_count += 1
            logger.debug(f"Hit Subtitle Frame! frame number = {count}")
            # Create Frame Capture Samples
            for i in range(1, frame_gap * sample_num):
                frame = cv2.resize(frame, (image_resize[1], image_resize[0]))
                if i % frame_gap == 0:
                    img_path = os.path.join(capture_save_path, folder_name + "_" + str(i//frame_gap) + ".jpg")
                    cv2.imwrite(img_path, frame)
                    
                #  Create Mask
                    if count in frame_subtitle_dict.keys():
                        Make_masks(os.path.join(mask_save_path, "mask_"+ str(i//frame_gap) + ".jpg")
                                        , frame_subtitle_dict[count])
                    else:
                        Make_masks(os.path.join(mask_save_path, "mask_"+ str(i//frame_gap) + ".jpg")
                                        , "", frame_has_no_sub=True)
                    mask_number += 1
                rval, frame = video.read()
                count = count + 1
                pbar.update(1)
        else:
            rval, frame = video.read()
            count = count+1
            pbar.update(1)
 
        
    video.release()
    logger.info(f"{movie} =====> cost time  : {t.time() - t1}")



def Inference_Extract_Video_info(args, movie, video_file):
    '''
    It's like the function Extract_Video_info(), but we don't have subtitle file while inference.
    So we just need to capture the frames and save them, not detect subtitle mask in this function.

    Argument:
        movie: str,
            movie name (folder name in MovieSubtitle_Dataset).
        video_file: str,
            the video file name with extension .avi, .mp4 and .mkv.
            

    Return
        Created samples folders  of this movie.
    '''
    sample_num = args.sample_num
    frame_gap = args.frame_gap
    image_resize = tuple(args.img_size)
    Scene_detection = args.scene_detection

    video_path = os.path.join(movie, video_file)
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    image_size = (int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(video.get(cv2.CAP_PROP_FRAME_WIDTH)))
    logger.info("{} => fps = {}, total_frames = {}, original image_size = {}, resize to {}".format(video_path, round(fps), total_frames, image_size, image_resize))


    # Scene Detection, if necessary.
    if Scene_detection:
        break_point_list = PyScene_detect(movie, video_file)

    # make Video Capture
    if video.isOpened():
        rval, frame = video.read()
    else:
        rval = False
        print("open video failed")

    # define path format
    count = 1    
    global mask_folder_count
    pbar = tqdm(total = total_frames)
    
    # Create Capture of Video
    folder_name_list = list()
    t1 = t.time()
    logger.info(f"{movie} : Creating samples folders ")
    for break_point_idx in range(len(break_point_list)-1):
        
        if break_point_list[break_point_idx+1] - break_point_list[break_point_idx] > frame_gap*sample_num:
            # Create Samples folder
            folder_name = randomString() + "_" + str(finished_movie_count)
            folder_name_list.append(folder_name)
            capture_save_path = os.path.join("MovieDataset", "Test", "videos", folder_name)
            if not os.path.exists(capture_save_path):
                os.makedirs(capture_save_path)
            
            for frame_number in range(break_point_list[break_point_idx], break_point_list[break_point_idx+1]):
                if (frame_number - break_point_list[break_point_idx]) % frame_gap == 0:
                    img_path = os.path.join(capture_save_path, folder_name + "_" + str((frame_number - break_point_list[break_point_idx])//frame_gap) + ".jpg")
                    frame = cv2.resize(frame, (image_resize[1], image_resize[0]))
                    logger.debug(f"save-> {frame_number} , at-> {img_path}")
                    cv2.imwrite(img_path, frame)

                rval, frame = video.read()
                count = count + 1
                pbar.update(1)
            
        else:
            for frame_number in range(break_point_list[break_point_idx], break_point_list[break_point_idx+1]):
                rval, frame = video.read()
                count = count + 1
                pbar.update(1)

   
    video.release()

    logger.info(f"There are {len(folder_name_list)} folders were created, list below: {folder_name_list}")
    logger.info(f"Inference {movie} =====> cost time  : {t.time() - t1}")
    return folder_name_list
    




def Process_Movie(args, subtitle_file_path_dict, movie_list, Inference = True):
    '''
        find the video file name of each movie, and then call functions to save captures, subtitle masks, etc.
    '''

    for movie in movie_list:
        # Start Process a Movie
        movie_files = os.listdir(movie)
        for file_name in movie_files:
            extension = file_name.split(".")[-1].lower()
            # found Movie video file          
            if extension == "avi" or extension == "mkv" or extension == "mp4":
                if Inference:
                    logger.info(f"Process_Movie: ./{movie}/{file_name}")
                    created_folder_list = Inference_Extract_Video_info(args, movie, file_name)
                    Detect_Text(created_folder_list)
                else:
                    Extract_Video_info(args, movie, file_name, subtitle_file_path_dict)
                    
        # Finish Processing a Movie
        global finished_movie_count 
        finished_movie_count += 1
   
        break
    # print(subtitle_file_path)



if __name__ == "__main__":
    args = get_parser()
    if args.clean_old_files:
        if os.path.exists("MovieDataset"):
            shutil.rmtree("MovieDataset", ignore_errors=True)
        if os.path.exists("CRAFT-pytorch-master/result"):
            shutil.rmtree("CRAFT-pytorch-master/result", ignore_errors=True)
        del_csv = list()
        for item in os.listdir("PySceneDetect-0.5.1.1"):
            if item.split(".")[-1] == "csv":
                del_csv.append(item)
        
        for item in del_csv:
            os.remove(f"PySceneDetect-0.5.1.1/{item}")
            print("del", item)
    exit(0)


    Inference = args.inference
    logger = Create_Logger(logging.INFO)
    # create Subtitle Text File
    subtitle_file_path_dict, movie_list = Check_SRT_File(Inference=Inference)
    print("here->", subtitle_file_path_dict, movie_list)
    # Parse subtitle text file content
    Process_Movie(args, subtitle_file_path_dict, movie_list, Inference=Inference)

    
    


