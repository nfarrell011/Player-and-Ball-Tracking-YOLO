"""
    Final Project: 
        Ball & Player Tracking; Team Differentiation
    Group Memebers:
        Joseph Nelson Farrell, Harshil Bhojwani, Leonardo DeCraca, & Priyanka Dipak Gujar
    CS 5330 Pattern Recognition & Computer Vision
    Professor Bruce Maxwell, PhD.
    Spring 2024

    This file can be used to clip to a video.
"""
from utils.utils_general import clip_video, track_ball
import sys
import os
from pathlib import Path

def main(argv):
    """
    This funtions will clip a video based on frame ranges.
    
    Parameters:
        argv (list): List of command-line arguments
            1. Raw file
            2. Output file
            3. Start frame
            4. End frame

    Returns:
        None
    """
    # pathing
    HOME = Path(os.getcwd())
    HOME_STR = str(HOME)

    # params check
    if len(argv) != 5:
        raise ValueError("Program needs: input file, output file, start frame, and end frame.!")
    else:
        raw_file = argv[1]
        clipped_file = argv[2]
        start_frame = int(argv[3])
        end_frame = int(argv[4])

    # clip video
    path_to_original = HOME_STR + "/data_raw/" + raw_file
    path_to_output = HOME_STR + "/data_raw/" +  clipped_file
    clip_video(path_to_original, path_to_output, start_frame, end_frame)
    
if __name__ == "__main__":
    main(sys.argv)