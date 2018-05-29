from __future__ import print_function
import cv2

def load_image(image_path):
    img = cv2.imread(image_path)
    return img
