"""
    Final Project: 
        Ball & Player Tracking; Team Differentiation
    Group Memebers:
        Joseph Nelson Farrell, Harshil Bhojwani, Leonardo DeCraca, & Priyanka Dipak Gujar
    CS 5330 Pattern Recognition & Computer Vision
    Professor Bruce Maxwell, PhD.
    Spring 2024

    This file will process a video clip. It will track the ball (using a Kalman filter), track players (people),
     and differentiate groupes of people (different teams and referee).
"""
# libraries
from pathlib import Path
import os
import sys
from ultralytics import YOLO
import ultralytics
import cv2
import matplotlib.pyplot as plt
import numpy as np

# modules
import utils.utils_label_teams_boolean_mask as ul

def main(argv):
    """
    This function will process a video clip. It will save a processed version of the video in the 
    'processed_videos' folder (if it does not exsit, one will be created). The processed video will track the ball, 
    players, and referees. The different teams and the referees will be differentiated and labeled.

    Parameters:
        argv (list): List of command-line arguments
            1. Raw file name
            2. Output file name 
            3. Color list (team jersey colors, and ref shirt color)
            4. Label list (team names and refs)
            5. Bool indicating if the program should run default YOLO on the entire clip.

    Returns:
        None
    """
    # pathing
    HOME = Path(os.getcwd())
    HOME_STR = str(HOME)
    FIGS_FOLDER = HOME_STR + "/figs/"
    RAW_VIDEO_FOLDER = HOME_STR + "/processed_videos/"

    # check folders
    if not os.path.exists(RAW_VIDEO_FOLDER):
        raise FileNotFoundError(f"The folder: '{RAW_VIDEO_FOLDER}' does not exist!")

    if not os.path.exists(FIGS_FOLDER):
        os.makedirs(FIGS_FOLDER, exist_ok=True)
        print(f"The folder '{FIGS_FOLDER}' was created.")
    else:
        print(f"The folder '{FIGS_FOLDER}' already exists.")


    SOURCE_VIDEO_PATH = f'{RAW_VIDEO_FOLDER}original_1_PROCESSED.mp4'

    # check source video
    if not os.path.exists(SOURCE_VIDEO_PATH):
        raise FileNotFoundError(f"The folder '{SOURCE_VIDEO_PATH}' does not exist!")

    # capture video
    cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)

    # check that it opened
    if not cap.isOpened():
        print(f'Error!! Video NOT opened!!')
    else:
        print(f'Video Opened!!')

    # iterate over the video
    i = 0
    while cap.isOpened():

        # get the frame
        ret, frame = cap.read()

        # check that frame was captured
        if not ret:
            print(f'Frame not captured!')
            break

        # get a keystroke
        key = cv2.waitKey(10)

        # if a key was pressed, update type and assess key, quit if 'q'
        if key != -1:
            key_char = chr(key)
            print(f'Key pressed: {key_char}')
            if key_char == 'q':
                print(f'Exiting program!!!')
                break
        if i == 141:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            plt.imshow(rgb_frame)
            plt.axis('off')
            #plt.title(f'Example Frame with Ball Detection', weight = 'bold', style = 'italic')
            plt.savefig(f'{HOME}/figs/ball_dedection_frame_example.jpg', bbox_inches = 'tight')
        
        if i == 9:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            plt.imshow(rgb_frame)
            plt.axis('off')
            #plt.title(f'Example Frame with No Ball Detection \n Kalman Filter', weight = 'bold', style = 'italic')
            plt.savefig(f'{HOME}/figs/kalman_filter_frame_example.jpg', bbox_inches = 'tight')
        
        if i == 197:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            plt.imshow(rgb_frame)
            plt.axis('off')
            #plt.title(f'Example Frame with No Ball Detection \n Prediction Terminated', weight = 'bold', style = 'italic')
            plt.savefig(f'{HOME}/figs/lost_ball_frame_example.jpg', bbox_inches = 'tight')
        i += 1

    # release cap and destroy windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(sys.argv)