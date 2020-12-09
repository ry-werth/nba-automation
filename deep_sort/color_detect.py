#================================================================
#
#   File name   : color_detect.py
#   Author      : Ryan Werth
#   Created date: 12-09-202
#   Description : code to detect color (team) of player
#
#================================================================

import cv2
import numpy as np

def find_color(roi, threshold=0.0):
  """
  Return the ratio of non white pixels to All pixels, white jersies should have low values
  """

  roi_hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
  # set a min and max for team colors
  COLOR_MIN = np.array([0, 0, 0])
  COLOR_MAX = np.array([255, 255, 100])

  # dark teams will remain with this mask
  mask = cv2.inRange(roi_hsv, COLOR_MIN, COLOR_MAX)
  res = cv2.bitwise_and(roi,roi, mask= mask)

  # dark teams should have a higher ratio
  tot_pix = roi.any(axis=-1).sum()
  color_pix = res.any(axis=-1).sum()
  ratio = color_pix/tot_pix

  return(ratio)
