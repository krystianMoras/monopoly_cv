from PIL.ExifTags import TAGS
from PIL import Image
from ipywidgets import Video
from pprint import pprint
import random
import glob
import sys
import os
import PIL
import cv2
import numpy as np
import matplotlib.pyplot as plt

# UTILITIES

def create_tracker(tracker_type):
    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD',
                     'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

    if tracker_type == 'BOOSTING':
        return cv2.TrackerBoosting_create()
    if tracker_type == 'MIL':
        return cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
        return cv2.TrackerKCF_create()
    if tracker_type == 'TLD':
        return cv2.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        return cv2.TrackerMedianFlow_create()
    if tracker_type == 'GOTURN':
        return cv2.TrackerGOTURN_create()
    if tracker_type == 'MOSSE':
        return cv2.TrackerMOSSE_create()
    if tracker_type == "CSRT":
        return cv2.TrackerCSRT_create()

def draw_bbox(frame, bbox, color=(255, 255, 255)):
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(frame, p1, p2, color, 2, 1)

def imshow(a):
    a = a.clip(0, 255).astype('uint8')
    if a.ndim == 3:
        if a.shape[2] == 4:
            a = cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA)
        else:
            a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    display(PIL.Image.fromarray(a))

def draw_rectangle_from_contours(source_img,contours):
    drawing = source_img.copy()
    for i in range(len(contours)):
        
        rect = cv2.minAreaRect(contours[i])
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        cv2.drawContours(drawing, [box], 0, (0, 0, 255), 2)
    return drawing

# DETECTION METHODS

def mask_color(frame,lower,upper):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, lower, upper)

def thresholds_for_crop(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    min_h, max_h = np.min(hsv,axis=(0,1)), np.max(hsv,axis=(0,1))
    return min_h, max_h

def filter_for_counter(mask, lower_bound, upper_bound):
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5)))
    
    mask = cv2.erode(mask,np.ones((5,5)))
    mask = cv2.dilate(mask,np.ones((5,5)))
    # for 1080p 7,7 worked okay, for 480p we tried 5,5 3,3 3,3 
    # imshow(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_filtered = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > lower_bound and area < upper_bound:
            contours_filtered.append(cnt)
        
    return contours_filtered, mask

def find_counter(source_img, contours, inverse = 0):
    counter_contour = None
    max_area = np.inf * (-1)
    if inverse != 0:
        max_area = np.inf
    for cnt in contours:
        area = cv2.contourArea(cnt)

        if inverse == 0:
            if area > max_area:
                counter_contour = cnt
        else:
            if area < max_area:
                counter_contour = cnt

    rect = cv2.minAreaRect(counter_contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # get width and height of the detected rectangle
    width = int(rect[1][0])
    height = int(rect[1][1])


    src_pts = box.astype("float32")
    # coordinate of the points in box points after the rectangle has been
    # straightened
    dst_pts = np.array([[0, height-1],
                        [0, 0],
                        [width-1, 0],
                        [width-1, height-1]], dtype="float32")

    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # directly warp the rotated rectangle to get the straightened rectangle
    warped = cv2.warpPerspective(source_img, M, (width, height))
    return warped, box

def corners_to_cvbbox(box):
    p1, p2, p3, p4 = box
    x1, y1, x2, y2, x3, y3, x4, y4 = p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], p4[0], p4[1]
    x_min = min(x1, x2, x3, x4)
    y_min = min(y1, y2, y3, y4)
    x_max = max(x1, x2, x3, x4)
    y_max = max(y1, y2, y3, y4)

    new_bbox = (x_min, y_min, x_max-x_min, y_max-y_min)
    return new_bbox

def approximate_cvbbox(box, more=5):
    x = box[0] - more
    y = box[1] - more
    w = box[2] + more
    h = box[3] + more

    return (x, y, w, h)

def get_player_box(frame, color_crop, lower_size, upper_size, inv=0, approx=0):
    min_h, max_h = thresholds_for_crop(color_crop)
    
    lower_col = np.array(min_h, np.uint8)
    upper_col = np.array(max_h, np.uint8)
    mask = mask_color(frame, lower_col ,upper_col)
    # imshow(mask)
    contours, mask2_ = filter_for_counter(mask, lower_size, upper_size)

    counter_crop, box = find_counter(frame, contours, inv)
    
    final_box = approximate_cvbbox(corners_to_cvbbox(box), approx)
    
    return final_box, counter_crop
