"""
    Final Project: 
        Ball & Player Tracking - Team Differentiation
    Group Memebers:
        Joseph Nelson Farrell, Harshil Bhojwani, Leonardo DeCraca, & Priyanka Dipak Gujar
    CS 5330 Pattern Recognition & Computer Vision
    Professor Bruce Maxwell, PhD.
    Spring 2024

    This file contains a library of functions that perfrom team differentiation using BGR
    color space.

    Functions:
        1. get_color_ranges
        2. get_player_team_color
"""
# libraries
import numpy as np
import cv2

########################################################################################################
# 1
def get_color_ranges(color_list: list) -> dict:
    """ 
        This function contains BGR values ranges for various colors, it will return a dict of colors needed
        to differentiate the teams defined by the input list.

        Parameters:
            color_list: (list) - A list of strings, color names. The colors of the jerseys of the teams.

        Returns:
            color_range_dict: (dict) - A dictionary of the colors (keys) and their BGR ranges (values).
    """

    color_ranges = {
                    'white': ([187, 169, 112], [255, 255, 255]),
                    'red': ([0, 0, 50], [100, 100, 255]),
                    'yellow': ([0, 100, 100], [10, 255, 255]),
                    'black': ([0, 0, 0], [50, 50, 50]),
                    'blue': ([43, 31, 4], [250, 88, 50]) 
                    }


    color_range_dict = {key: color_ranges[key] for key in color_list if key in color_ranges}

    return color_range_dict

########################################################################################################
# 2
def get_player_team_color(player_region: np.ndarray, color_range_dict: dict, color_list: list) -> tuple:
    """ 
        This function will take the color_range_dict and the player bounding boxes and generate a thresholded image where the color
        of interest remains and all other pixels are black. It will compute a ratio of color pixels versus total pixels in the 
        bounding box and put the ratios in a list. The argmax of this list will be the team color of the current player.

        Parameters:
            player_region: (np.ndarray) - Omage region of a player, defined by a bounding box.
            color_range_dict: (dict) - A dictionary of the colors (keys) and their BGR ranges (values).
            color_list: (list) - A list of strings, color names. The colors of the jerseys of the teams.

        Returns:
            color: (str) - The color of the jersey of the current player.
            masked_images: (dict) - This is a dict of the thresholded images as the values, and color of interest as
                as the key. Each player will have three. This is used to troubleshoot and test ONLY.
    """
    # this is for displaying and not really used
    masked_images = {}

    # this will store the ratios color pixels versus black pixels
    mask_ratios_list = []

    # iterate over the color dict; color ranges
    for color, (lower, upper) in color_range_dict.items():

        # convert lower and upper bounds to NumPy arrays
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        
        # create mask for the current color
        mask = cv2.inRange(player_region, lower, upper)
        
        # apply mask to player region
        masked_image = cv2.bitwise_and(player_region, player_region, mask=mask)
        
        # save the masked image; for display ONLY
        masked_images[color] = masked_image

        # get the dims of the region
        height, width, _ = masked_image.shape

        # get the total pixels
        total_pix = height * width

        # black pixel tracker
        black_pix = 0

        # iterate over region; tracking num black pixels
        for i in range(height):
            for j in range(width):
                pixel = masked_image[i, j]
                pix_sum = np.sum(pixel)
                if pix_sum == 0:
                    black_pix += 1

        # compute ratio of color pixels to total pixels
        ratio = (total_pix - black_pix) / total_pix

        # add ration of current mask to ratio list
        mask_ratios_list.append(ratio)

    # get the max ratio
    # this is detected color; used to differentiate the teams   
    max_idx = np.argmax(mask_ratios_list)

    # get the corresponding color
    color = color_list[max_idx]

    return color, masked_images



