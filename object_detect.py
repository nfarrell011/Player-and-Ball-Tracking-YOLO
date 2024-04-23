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
    PROCESSED_FRAMES_FOLDER = HOME_STR + "/processed_videos/"
    RAW_VIDEO_FOLDER = HOME_STR + "/data_raw/"

    # check folders
    if not os.path.exists(RAW_VIDEO_FOLDER):
        raise FileNotFoundError(f"The folder: '{RAW_VIDEO_FOLDER}' does not exist!")

    if not os.path.exists(PROCESSED_FRAMES_FOLDER):
        os.makedirs(PROCESSED_FRAMES_FOLDER, exist_ok=True)
        print(f"The folder '{PROCESSED_FRAMES_FOLDER}' was created.")
    else:
        print(f"The folder '{PROCESSED_FRAMES_FOLDER}' already exists.")

    # params check
    if len(argv) != 6:
        print()
        print(f"The program is continuing with default values. Use sys.args to adjust. See README.")
        print()
        PROCESSED_FRAMES_FILE_PATH = PROCESSED_FRAMES_FOLDER + "original_PROCESSED.mp4"
        SOURCE_VIDEO_PATH = f'{RAW_VIDEO_FOLDER}original_1.mp4'
        color_list = ["white", "red", "yellow"]
        person_cat_list = ['MCI', 'RMA', 'Ref']
        run_full_clip = False
    else:
        SOURCE_VIDEO_PATH = RAW_VIDEO_FOLDER + sys.argv[1]
        PROCESSED_FRAMES_FILE_PATH = PROCESSED_FRAMES_FOLDER + sys.argv[2]
        color_list = list(sys.argv[3])
        person_cat_list = list(sys.argv[4])
        run_full_clip = bool(sys.argv[5])
    
    # check source video
    if not os.path.exists(SOURCE_VIDEO_PATH):
        raise FileNotFoundError(f"The folder '{SOURCE_VIDEO_PATH}' does not exist!")

    # import the model ~ YOLOv8x
    ultralytics.checks()
    MODEL = "yolov8x.pt"
    model = YOLO(MODEL)
    model.fuse()

    # parameters
    ball_class_id = 32
    person_class_id = 0
    show_masked_images = False
    generate_mask_examples_figs = True
    num_mask_example_figs = 0

    ############################################################################################################
    #################### This will run YOLO on the full video, if 'run_full_clip' parameter is TRUE ############
    if run_full_clip:
        os.system("yolo task=detect mode=predict model=yolov8x.pt source=" + SOURCE_VIDEO_PATH)
    ############################################################################################################
    ############################################################################################################

    # initialize Kalman filter
    kalman = cv2.KalmanFilter(4, 2)  # 4 dimensions (x, y, dx, dy), 2 measurements (x, y)

    # initialize state
    kalman.statePre = np.array([0, 0, 0, 0], dtype = np.float32)

    # define transition matrix
    kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], dtype = np.float32)

    # define measurement matrix:
    kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                        [0, 1, 0, 0]], dtype = np.float32)

    # capture video
    cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)

    # check that it opened
    if not cap.isOpened():
        print(f'Error!! Video NOT opened!!')
    else:
        print(f'Video Opened!!')

    # hypers for saving frames to mp4 file
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

    # list to save processed frames
    processed_frames_list = []

    # this will be used to track the number of consecutive frames without a ball detection
    frames_without_ball = 0

    # this is used to activate the Kalman
    first_ball_detect = False

    # iterate over the video
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

        # execute YOLO
        results = model(frame)

        # detection bounding boxes
        bounding_boxes = results[0].boxes.xyxy.numpy()

        # detection classes
        classes = results[0].boxes.cls.numpy()

        # detection confidences
        confidences = results[0].boxes.conf.numpy()

        # check of the ball was detected
        if ball_class_id in classes:
            first_ball_detect = True
            ball_detect = True
            frames_without_ball = 0
        else:
            ball_detect = False
            frames_without_ball += 1

        if first_ball_detect:

            # this is used to skip the ball if it been already annotated
            ball_idx = None

            # if the ball was detected, update Kalman with measurement
            if ball_detect:

                # the index of the ball from the classes array
                ball_idx = np.argwhere(classes == ball_class_id)

                # get the ball bounding box
                ball_bounding_box = bounding_boxes[ball_idx]
                ball_bounding_box = ball_bounding_box[0][0]

                # get the ball confidence
                ball_confidence = confidences[ball_idx]

                # get ball coordinates, put in for
                x = int(ball_bounding_box[0])
                y = int(ball_bounding_box[1])
                w = int(ball_bounding_box[2]) - int(ball_bounding_box[0])
                h = int(ball_bounding_box[3] - ball_bounding_box[1])

                # get the ball measurement for Kalman filter
                measurement = np.array([[x + w / 2], [y + h / 2]], dtype = np.float32)

                # make prediction and update Kalman
                prediction = kalman.predict()
                kalman.correct(measurement)

                # draw bounding box
                start_point = (int(ball_bounding_box[0]), int(ball_bounding_box[1]))
                end_point = (int(ball_bounding_box[2]), int(ball_bounding_box[3]))
                color = (255, 0, 0) 
                thickness = 1
                cv2.rectangle(frame, start_point, end_point, color, thickness)

                # put text
                text = f'Ball Measurement (Detected)'
                font = cv2.FONT_HERSHEY_COMPLEX
                font_scale = 0.4
                start_point = (x + 15, y + 10)
                color = (255, 0, 0)
                thickness = 1
                line_type = cv2.LINE_AA
                cv2.putText(frame, text, start_point, font, font_scale, color, thickness, line_type)
                
            # Kalman prediction
            prediction = kalman.predict()

            # if the ball was not detected, draw the predicted box based on Kalman
            if not ball_detect:

                # check if the number frame without detection is below threshold
                if frames_without_ball <= 10:

                    # get the predictions
                    predicted_x, predicted_y = prediction[:2]

                    # hypers for putText() and rectangle
                    text = f'Kalman Prediction'
                    font = cv2.FONT_HERSHEY_COMPLEX
                    font_scale = 0.5
                    start_pt = (int(predicted_x) + 10, int(predicted_y) + 10)
                    color = (0, 255, 0)
                    thickness_text = 1
                    line_type = cv2.LINE_AA
                    start_point = (int(predicted_x - w / 2), int(predicted_y - h / 2))
                    end_point = (int(predicted_x + w / 2), int(predicted_y + h / 2))
                    thickness_rect = 1

                    # draw rect
                    cv2.rectangle(frame, start_point, end_point, color, thickness_rect)

                    # put text
                    cv2.putText(frame, text, start_pt, font, font_scale, color, thickness_text, line_type)
                else:
                    text = f'The BALL has been lost!! Prediction Terminated!!'
                    font = cv2.FONT_HERSHEY_COMPLEX
                    font_scale = 1
                    start_pt = (200, 600)
                    color = (255, 255, 255)
                    thickness = 1
                    line_type = cv2.LINE_AA
                    thickness = 1
                    cv2.putText(frame, text, start_pt, font, font_scale, color, thickness, line_type)
        
        # this will remove all detections except for people (players)
        condition = classes == person_class_id
        bounding_boxes = bounding_boxes[condition]
        classes = classes[condition]

        # this will iterate over the rest of the bounding boxes
        #for index, (box, id) in enumerate(zip(bounding_boxes, classes)):
        for box in bounding_boxes:

            player_region = frame[int(box[1]): int(box[3]),
                    int(box[0]): int(box[2])]
            
            color_range_dict = ul.get_color_ranges(color_list)

            color, masked_images = ul.get_player_team_color(player_region, color_range_dict, color_list)

            # this will show the masked images if activated
            if show_masked_images == True:
                for color_2, masked_image in masked_images.items():
                    cv2.imshow(color_2.capitalize(), masked_image)  
                cv2.waitKey(0) 
                cv2.destroyAllWindows()

            # this generate figs of the masked examples and save them to figs
            if generate_mask_examples_figs == True:
                if num_mask_example_figs < 5:
                    fig, axes = plt.subplots(1, len(masked_images), figsize=(8, 5))
                    fig.suptitle("Masking Outcome for Player Bounding Box \n BGR Color Space", 
                                 weight = "bold", 
                                 fontsize = 16,
                                 y = 1.05)
                    for ax, (color_, masked_image) in zip(axes, masked_images.items()):
                        ax.imshow(masked_image)
                        ax.set_title(f'{color_.capitalize()} Mask')  
                        ax.axis('off')
                    num_mask_example_figs += 1
                    plt.savefig(f'{HOME}/figs/masking_example_BGR_{num_mask_example_figs}.jpg', bbox_inches = 'tight')

            # set the color and text for each bounding box
            if color == color_list[0]:                                  
                color = (255, 255, 255)
                text = person_cat_list[0]
            elif color == color_list[1]:
                color = (0, 0, 255)
                text = person_cat_list[1]
            else:
                color = (0, 255, 255)
                text = person_cat_list[2]

            # draw bounding box
            start_point = (int(box[0]), int(box[1]))
            end_point = (int(box[2]), int(box[3]))
            thickness = 1
            cv2.rectangle(frame, start_point, end_point, color, thickness)

            # put text
            font = cv2.FONT_HERSHEY_COMPLEX
            font_scale = 0.5
            start_pt = (start_point[0], start_point[1] - 10)
            thickness = 1
            line_type = cv2.LINE_AA
            cv2.putText(frame, text, start_pt, font, font_scale, color, thickness, line_type)

            cv2.imshow('Frame', frame)

        # add frame to processed frames list
        processed_frames_list.append(frame)

        # update ball_detect
        ball_detect = False

    # release cap and destroy windows
    cap.release()
    cv2.destroyAllWindows()

    # write frames to output file
    out = cv2.VideoWriter(PROCESSED_FRAMES_FILE_PATH, cv2.VideoWriter_fourcc(*'avc1'), frame_rate, (frame_width, frame_height))
    for frame in processed_frames_list:
        out.write(frame)

    # Release the video writer object
    out.release()

if __name__ == "__main__":
    main(sys.argv)


