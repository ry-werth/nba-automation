#================================================================
#
#   File name   : object_tracker.py
#   Author      : PyLessons
#   Created date: 2020-09-17
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#   Description : code to track detected object from video or webcam
#
#================================================================
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import tensorflow as tf
from yolov3.utils import Load_Yolo_model, image_preprocess, postprocess_boxes, nms, draw_bbox, read_class_names
from yolov3.configs import *
import time

from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet
from deep_sort.color_detect import find_color
from scipy import stats


# The main function for analyzing our videos
def Object_tracking(Yolo, video_path, output_path, input_size=416, show=False, CLASSES=YOLO_COCO_CLASSES, score_threshold=0.3, iou_threshold=0.45, rectangle_colors='', Track_only = [], custom_yolo=None, custom_classes=YOLO_CUSTOM_CLASSES, Custom_track_only=[]):
    # Definition of the parameters
    max_cosine_distance = 0.7
    nn_budget = None


    #initialize deep sort object
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    times, times_2 = [], []

    if video_path:
        vid = cv2.VideoCapture(video_path) # detect on video
    else:
        vid = cv2.VideoCapture(0) # detect from webcam

    # by default VideoCapture returns float instead of int
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, codec, fps, (width, height)) # output_path must be .mp4

    NUM_CLASS = read_class_names(CLASSES)
    key_list = list(NUM_CLASS.keys())
    val_list = list(NUM_CLASS.values())

    # set a bunch of flags and variables for made baskets and possessions
    possession = None
    possession_list = []
    combined_possession_avg = 0.5

    total_basket_count=0
    basket_frame_list = []

    baskets_dict = {"Dark": 0, "Light": 0}

    made_basket_first_frame = 0
    made_basket_frames = 0
    basket_marked = False

    if custom_yolo:
      NUM_CUSTOM_CLASS = read_class_names(custom_classes)
      custom_key_list = list(NUM_CUSTOM_CLASS.keys())
      custom_val_list = list(NUM_CUSTOM_CLASS.values())

    frame_counter = 0
    # loop through each frame in video
    while True:
        _, frame = vid.read()

        try:
            first_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            original_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
            frame_counter += 1
        except:
            break

        image_data = image_preprocess(np.copy(first_frame), [input_size, input_size])
        #image_data = tf.expand_dims(image_data, 0)
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        t1 = time.time()
        # CUSTOM BLOCK FOR BASKETBALL
        if custom_yolo:

          if YOLO_FRAMEWORK == "tf":
            # use yolo model to make prediction on the image data
            custom_pred_bbox = custom_yolo.predict(image_data)

          # reshape our data to be in correct form for processing
          custom_pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in custom_pred_bbox]
          custom_pred_bbox = tf.concat(custom_pred_bbox, axis=0)

          # get boxes based on threshhold
          custom_bboxes = postprocess_boxes(custom_pred_bbox, original_frame, input_size, 0.3)
          # custom_bboxes = nms(custom_bboxes, iou_threshold, method='nms')

          # extract bboxes to boxes (x, y, width, height), scores and names
          custom_boxes, custom_scores, custom_names = [], [], []
          for bbox in custom_bboxes:
              if len(Custom_track_only) !=0 and NUM_CUSTOM_CLASS[int(bbox[5])] in Custom_track_only or len(Custom_track_only) == 0:
                  custom_boxes.append([bbox[0].astype(int), bbox[1].astype(int), bbox[2].astype(int)-bbox[0].astype(int), bbox[3].astype(int)-bbox[1].astype(int)])
                  custom_scores.append(bbox[4])
                  custom_names.append(NUM_CUSTOM_CLASS[int(bbox[5])])

          # Obtain all the detections for the given frame.
          custom_boxes = np.array(custom_boxes)
          custom_names = np.array(custom_names)
          custom_scores = np.array(custom_scores)

          # take note of the highest "scoring" made basket and basketball obj in each frame
          highest_scoring_basketball = 0
          basketball_box = None
          basketball_center = None
          highest_scoring_made_basket = 0
          made_basket_box = None
          for i, bbox in enumerate(custom_bboxes):
            # loop through each bounding box to get the "best one" of the frame
            # we do this because sometimes our model will detect two, and we know there can only be one
            name = custom_names[i]
            score = round(custom_scores[i], 3)
            if name == 'basketball':
              if score > highest_scoring_basketball:
                highest_scoring_basketball = score
                basketball_box = bbox
            if name == 'made-basket':
              if score > .85 and score > highest_scoring_made_basket:
                highest_scoring_made_basket = score
                made_basket_box = bbox

          # if it sees a basketball, put a box on it and note the center (for possession)
          if basketball_box is not None:
            cv2.rectangle(original_frame, (int(basketball_box[0]), int(basketball_box[1])), (int(basketball_box[2]), int(basketball_box[3])), (0,0,255), 1)
            cv2.rectangle(original_frame, (int(basketball_box[0]), int(basketball_box[1]-30)), (int(basketball_box[0])+(10)*17, int(basketball_box[1])), (0,0,255), -1)
            cv2.putText(original_frame, "basketball" + "-" + str(highest_scoring_basketball),(int(basketball_box[0]), int(basketball_box[1]-10)),0, 0.5, (255,255,255),1)
            basketball_center = ( (basketball_box[2]+basketball_box[0])/2, (basketball_box[3]+basketball_box[1])/2 )


          if made_basket_box is not None:
            # if theres a made basket put the box on it
            cv2.rectangle(original_frame, (int(made_basket_box[0]), int(made_basket_box[1])), (int(made_basket_box[2]), int(made_basket_box[3])), (0,255,0), 1)
            cv2.rectangle(original_frame, (int(made_basket_box[0]), int(made_basket_box[1]-30)), (int(made_basket_box[0])+(15)*17, int(made_basket_box[1])), (0,255,0), -1)
            cv2.putText(original_frame, "made-basket" + " " + str(highest_scoring_made_basket),(int(made_basket_box[0]), int(made_basket_box[1]-10)),0, 0.6, (0,0,0),1)

            if made_basket_frames == 0:
              # if this is the first frame in the sequence
              made_basket_first_frame = frame_counter

            # increment a counter for made basket frames
            made_basket_frames += 1

            # if there were 3 consecuative frames AND we havnt marked the basket yet then lets count it!
            if made_basket_frames >= 3 and not basket_marked:
              basket_marked = True
              basket_frame_list.append(made_basket_first_frame)
              if possession:
                # record which "team" scored the basket
                baskets_dict[possession] += 1

          # if no made basket make sure the made basket counter is at zero
          else:
            # no made basket
            made_basket_frames = 0

          # 60 frames after a made basket we can reset the "marked basket" flag to False
          # in essence this means we start looking for made baskets again
          if basket_marked and frame_counter > basket_frame_list[-1] + 60:
            basket_marked = False

        # END CUSTOM BLOCK

        # PRESON PREDICTION and TRACKING BLOCK

        if YOLO_FRAMEWORK == "tf":
            pred_bbox = Yolo.predict(image_data)
        elif YOLO_FRAMEWORK == "trt":
            batched_input = tf.constant(image_data)
            result = Yolo(batched_input)
            pred_bbox = []
            for key, value in result.items():
                value = value.numpy()
                pred_bbox.append(value)

        t2 = time.time()

        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)

        bboxes = postprocess_boxes(pred_bbox, original_frame, input_size, score_threshold)
        bboxes = nms(bboxes, iou_threshold, method='nms')



        # extract bboxes to boxes (x, y, width, height), scores and names
        boxes, scores, names = [], [], []
        for bbox in bboxes:
            if len(Track_only) !=0 and NUM_CLASS[int(bbox[5])] in Track_only or len(Track_only) == 0:
                w = bbox[2].astype(int)-bbox[0].astype(int)
                h = bbox[3].astype(int)-bbox[1].astype(int)
                if h < height/3 and w < width/4:
                  if h > 120:
                    boxes.append([bbox[0].astype(int), bbox[1].astype(int), w, h])
                    scores.append(bbox[4])
                    names.append(NUM_CLASS[int(bbox[5])])

        # Obtain all the detections for the given frame.
        boxes = np.array(boxes)
        names = np.array(names)
        scores = np.array(scores)

        # detect jersey color using the tracked persons bounding box
        patches = [gdet.extract_image_patch(frame, box, [box[3], box[2]]) for box in boxes]
        color_ratios = [find_color(patch) for patch in patches]

        features = np.array(encoder(original_frame, boxes))

        # mark the detection
        detections = [Detection(bbox, score, class_name, feature, color_ratio) for bbox, score, class_name, feature, color_ratio in zip(boxes, scores, names, features, color_ratios)]

        # Pass detections to the deepsort object and obtain the track information.
        tracker.predict()
        tracker.update(detections)

        # Obtain info from the tracks
        tracked_bboxes = []
        color_ratio_list = []
        check_possession = False
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 5:
                continue

            color_ratio = track.get_color_ratio()
            color_ratio_list.append(color_ratio)

            bbox = track.to_tlbr() # Get the corrected/predicted bounding box
            class_name = track.get_class() #Get the class name of particular object
            tracking_id = track.track_id # Get the ID for the particular track
            index = key_list[val_list.index(class_name)] # Get predicted object index by object name

            tracked_bboxes.append(bbox.tolist() + [tracking_id, index]) # Structure data, that we could use it with our draw_bbox function

            # if there is a basketball in the frame, and its "in" a ersons bounding box, check what box it is in for psosession
            if basketball_center:
              if basketball_center[0] >= bbox[0] and basketball_center[0] <= bbox[2]:
                if basketball_center[1] >= bbox[1] and basketball_center[1] <= bbox[3]:
                  check_possession = True
                  if color_ratio <= .2:
                    # light team
                    possession_list.append(0)
                  else:
                    # dark team
                    possession_list.append(1)

            else:
              # no basketball in frame
              # possession_list.append(-1)
              # test_list.pop(0)
              pass

        # if the ball is in a bounding box, update out possession tracker
        if check_possession:
          if len(possession_list) > 60:
            # this function takes an average of the last 60 posessions marked to determine current position
            # it weights the most recent detections more
            # this algo is a WIP
            possession_list = possession_list[-60:]
            # full_avg = sum(possession_list)/len(possession)
            last_60_avg = sum(possession_list[-60:])/60
            last_30_avg = sum(possession_list[-30:])/30
            last_15_avg = sum(possession_list[-15:])/15
            last_5_avg = sum(possession_list[-5:])/5

            combined_possession_avg = round((last_60_avg + last_30_avg + last_15_avg + last_5_avg)/4,3)

            #most_common_possession = stats.mode(possession_list)[0]

          else:
            combined_possession_avg = round(sum(possession_list)/len(possession_list),3)

          # use our possession average to determine who has the ball right now
          if combined_possession_avg < 0.5:
            possession = "Light"
          elif combined_possession_avg > 0.5:
            possession = "Dark"


        # draw detection on frame
        image = draw_bbox(original_frame, tracked_bboxes, color_ratios=color_ratio_list, CLASSES=CLASSES, tracking=True)

        t3 = time.time()
        times.append(t2-t1)
        times_2.append(t3-t1)

        times = times[-20:]
        times_2 = times_2[-20:]

        ms = sum(times)/len(times)*1000
        fps = 1000 / ms
        fps2 = 1000 / (sum(times_2)/len(times_2)*1000)

        if possession == "Light":
          image = cv2.putText(image, "Posession: {}".format(possession), (width-400, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (50, 255, 255), 2)
        else:
          image = cv2.putText(image, "Posession: {}".format(possession), (width-400, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

        # image = cv2.putText(image, "Light: {} Dark: {} None: {}".format(possession_list.count(0), possession_list.count(1), possession_list.count(-1)), (400, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        image = cv2.putText(image, "Posession Avg: {}".format(combined_possession_avg), (400, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

        image = cv2.putText(image, "Time: {:.1f} FPS".format(fps), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

        # draw original yolo detection
        #image = draw_bbox(image, bboxes, CLASSES=CLASSES, show_label=False, rectangle_colors=rectangle_colors, tracking=True)

        print("Time: {:.2f}ms, Detection FPS: {:.1f}, total FPS: {:.1f}".format(ms, fps, fps2))
        if output_path != '': out.write(image)
        if show:
            cv2.imshow('output', image)

            if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break

    cv2.destroyAllWindows()
    return_data = {"baskets_dict": baskets_dict, "basket_frame_list": basket_frame_list}
    print("video saved to {}".format(output_path))
    return(return_data)


# yolo = Load_Yolo_model()
# Object_tracking(yolo, video_path, "/mydrive/basketball-videos/made-basket-tracker/game_3_person_tracking_detected.mp4", input_size=YOLO_INPUT_SIZE, show=False, iou_threshold=0.1, rectangle_colors=(255,0,0), Track_only = ["person"])
