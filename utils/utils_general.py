"""
    Final Project: 
        Ball & Player Tracking - Team Differentiation
    Group Memebers:
        Joseph Nelson Farrell, Harshil Bhojwani, Leonardo DeCraca, & Priyanka Dipak Gujar
    CS 5330 Pattern Recognition & Computer Vision
    Professor Bruce Maxwell, PhD.
    Spring 2024

    This file contains a functiont that will clip a video.

    Functions:
        1. clip_video
        2. track_ball
"""
# libraries
import cv2

########################################################################################################
# 1
def clip_video(path_to_source: str, save_path: str, clip_start: int, clip_end: int) -> None:
    cap = cv2.VideoCapture(path_to_source)
    if not cap.isOpened():
        raise IOError("Cannot open video file")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    width = width if width % 2 == 0 else width - 1
    height = height if height % 2 == 0 else height - 1

    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (width, height))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if clip_start <= frame_idx <= clip_end:
            out.write(frame)
        elif frame_idx > clip_end:
            break
        frame_idx += 1

    cap.release()
    out.release()

########################################################################################################
# 2
def track_ball(x1, y1, x2, y2, frame, ball_track_history):
   
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    ball_track_history.append((center_x, center_y))

        
    ball_track_history = ball_track_history[-10:]

    if len(ball_track_history) > 1:
        for i in range(1, len(ball_track_history)):
            cv2.line(frame, ball_track_history[i - 1], ball_track_history[i], (0, 255, 255), 2)




