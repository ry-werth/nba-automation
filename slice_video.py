#================================================================
#
#   File name   : slice_video.py
#   Author      : Ryan Werth
#   Created date: 12-09-202
#   Description : creates time stamp windows and slices up videos
#
#================================================================

import numpy as np
import cv2


def create_time_stamp_windows(made_basket_start_frames, seconds_before, seconds_after, fps):
    """
    feed in a list of frames and how many seconds before and after as well as fps
    returns a dictionary with all the start frames and all the end frames for sliceing
    """

    start_frames = np.array(made_basket_start_frames) - (seconds_before*fps)
    end_frames = np.array(made_basket_start_frames) + (seconds_after*fps)

    return_data = {"start_frames": start_frames, "end_frames": end_frames}
    return(return_data)


def video_slicer(filepath, save_path, start_indexes, end_indexes):
    """

    takes in a video path, a save path, start and end indexes
    saves a video thats cut at the specific indexes
    """
    cap = cv2.VideoCapture(filepath)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(save_path,cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width, frame_height))

    if len(start_indexes) != len(end_indexes):
        return("Time stamps must be the same length")

    basket_counter = 0
    frame_counter = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            # if no more frames then break
            break

        if basket_counter >= len(end_indexes):
            # if we've gotten all our slices then break
            break

        if frame_counter >= start_indexes[basket_counter] and frame_counter <= end_indexes[basket_counter]:
            # if we are in a basket then save that frame
            out.write(frame)
        elif frame_counter > end_indexes[basket_counter]:
            # if we just left a basket then increment our bascket counter
            basket_counter += 1

        frame_counter += 1


    cap.release()
    out.release()
    print("{} clips were sliced".format(str(basket_counter)))
    print("File Saved to {}".format(save_path))
