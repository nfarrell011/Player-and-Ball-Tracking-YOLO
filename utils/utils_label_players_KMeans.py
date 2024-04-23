"""
    Final Project: 
        Ball & Player Tracking - Team Differentiation
    Group Memebers:
        Joseph Nelson Farrell, Harshil Bhojwani, Leonardo DeCraca, & Priyanka Dipak Gujar
    CS 5330 Pattern Recognition & Computer Vision
    Professor Bruce Maxwell, PhD.
    Spring 2024

    This file contains a library of functions that perfrom team differentiation KMEANS

    These functions are NOT used in the current setup.

    Functions:
        1. cluster_k_means
        2. get_player_color
        3. assign_team_color
        4. assign_player_to_team
    
    This is approach is based on a similar approach performed here:
        https://github.com/abdullahtarek/football_analysis

"""
from sklearn.cluster import KMeans

def cluster_k_means(image):

    # reshape image into 2d array
    image_2d = image.reshape(-1, 3)

    # perform k-means with k = 2
    k_means = KMeans(n_clusters = 2,
                    init = "k-means++",
                    n_init = 1)
    
    # fit model
    k_means.fit(image_2d)

    return k_means

def get_player_color(frame, bounding_box):

    # get the player bounding box of the image
    player = frame[int(bounding_box[1]): int(bounding_box[3]),
                    int(bounding_box[0]): int(bounding_box[2])]
    
    # get the top half, i.e., the jersey
    top_half = player[0: int(player.shape[0] / 2), :]

    top_half = player

    # perform KMeans
    k_means = cluster_k_means(top_half)

    # get cluser labels for each pixel
    labels = k_means.labels_

    # reshape cluster labels
    clustered_player = labels.reshape(top_half.shape[0], top_half.shape[1])

    # get player cluster
    corner_clusters = [clustered_player[0, 0], clustered_player[0, -1], 
                       clustered_player[-1, 0], clustered_player[-1, -1]]
    
    non_player_cluster = max(set(corner_clusters), key = corner_clusters.count)
    player_cluster = 1 - non_player_cluster
    player_color = k_means.cluster_centers_[player_cluster]

    return player_color

def assign_team_color(frame, bounding_boxes):

    player_colors = []

    for player_bounding_box in bounding_boxes:
        player_color = get_player_color(frame, player_bounding_box)
        player_colors.append(player_color)

    k_means = KMeans(n_clusters = 2, init = "k-means++", n_init = 1)
    k_means.fit(player_colors)

    team_colors_dict = {"team_1": k_means.cluster_centers_[0],
                        "team_2": k_means.cluster_centers_[1]}
    
    return team_colors_dict, k_means

def assign_player_to_team(frame, player_bounding_box, k_means):

    player_color = get_player_color(frame, player_bounding_box)
    team_id = k_means.predict(player_color.reshape(1, -1))
    team_id += 1
    return team_id






