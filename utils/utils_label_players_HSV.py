"""
    Final Project: 
        Ball & Player Tracking - Team Differentiation
    Group Memebers:
        Joseph Nelson Farrell, Harshil Bhojwani, Leonardo DeCraca, & Priyanka Dipak Gujar
    CS 5330 Pattern Recognition & Computer Vision
    Professor Bruce Maxwell, PhD.
    Spring 2024

    This file contains a library of functions that perfrom team differentiation using HSV
    color space and histograms.

    These functions are NOT used in the current setup.

    Functions:
        1. generate_HSV_hist_feature_vector
        2. get_region
        3. get_team_feature_vectors
        4. cosine distance
"""
import cv2
import numpy as np

########################################################################################################
# 1
def generate_HSV_hist_feature_vector(player_regoin):
    """
        Generate a HSV histogram feature vector out of a image region

        Parameters:
            player_region: (np.ndarray) - the region of the frame containing a player.
        
        Returns:
            feature_vector: (np.ndarray) - an HSV color space feature vector.
    """

    hsv_region = cv2.cvtColor(player_regoin, cv2.COLOR_BGR2HSV)

    # compute histograms for each HSV channel
    h_hist = cv2.calcHist([hsv_region], [0], None, [256], [0, 256]).flatten()
    s_hist = cv2.calcHist([hsv_region], [1], None, [256], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv_region], [2], None, [256], [0, 256]).flatten()

    # concatenate histograms into a single feature vector
    feature_vector = np.concatenate((h_hist, s_hist, v_hist))

    return feature_vector

########################################################################################################
# 2
def get_region(player_region, region_size):
    """ 
        Get a square crop of the middle of a region.

        Parameters:
            player_region: (np.ndarray) - the region of the frame containing a player.
            region_size: (int) - the dims of the crop.
        
        Returns:
            player: (np.ndarray) - the cropped player region of the frame.
    """

    # calc the start and end coordinates for the middle region
    middle_region_start_x = (player_region.shape[1] - region_size) // 2
    middle_region_start_y = (player_region.shape[0] - region_size) // 2
    middle_region_end_x = middle_region_start_x + region_size
    middle_region_end_y = middle_region_start_y + region_size

    # extract the middle 20x20 pixel region
    player = player_region[middle_region_start_y:middle_region_end_y,
                                middle_region_start_x:middle_region_end_x]
    
    return player

########################################################################################################
# 3
def get_team_feature_vectors(player_bounding_boxes, frame):
    """ 
        Get HSV histogram feature vectors for each team.

        Parameters:
            player_bounding_boxes: (list) - a list of the player bounding boxes.
            frame: (np.ndarray) - the current frame
        
        Returns:
            min_player_feature_vector: (np.ndarray) - team 1 feature vector
            max_player_feature_vector: (np.ndarray) - team 1 feature vector 
    """

    # define the ROI dimensions (it will be sqaure)
    region_size = 20

    # list to store bounding box means
    bounding_box_means = []

    # list to store the regions
    player_regions = []

    # iterate over the bounding boxes
    for bounding_box in player_bounding_boxes:
        
        # get the player region of the frame
        player_region = frame[int(bounding_box[1]): int(bounding_box[3]),
                    int(bounding_box[0]): int(bounding_box[2])]
        
        player = get_region(player_region, region_size)

        # get the mean pixel value of the region
        player_mean = np.mean(player)

        # append lists
        bounding_box_means.append(player_mean)
        player_regions.append(player)

    # get the min and max of bounding box means
    max_mean = np.argmax(bounding_box_means)

    min_mean = np.argmin(bounding_box_means)
    bounding_box_means[min_mean] = np.inf
    min_mean = np.argmin(bounding_box_means)

    # get the corresponding player regions; 20x20
    max_player = player_regions[max_mean]
    min_player = player_regions[min_mean]

    cv2.imshow("max_player", max_player)
    cv2.imshow("min_player", min_player)
    cv2.waitKey(0)

    # get the hsv feature vectors
    min_player_feature_vector = generate_HSV_hist_feature_vector(min_player)
    max_player_feature_vector = generate_HSV_hist_feature_vector(max_player)

    return min_player_feature_vector, max_player_feature_vector

########################################################################################################
# 4
def cosine_distance(v1, v2):
    """
    Compute cosine distance between two vectors.
    
    Parameters:
        v1: (np.ndarray) - first vector.
        v2: (np.ndarray) - second vector.
    
    Returns:
        cosine_distance: (float) - distance between the two vectors.
    """
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    cosine_similarity = dot_product / (norm_v1 * norm_v2)
    cosine_distance = 1 - cosine_similarity
    
    return cosine_distance
