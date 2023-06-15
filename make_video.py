import os

import cv2

img_array = []
for filename in os.listdir("./*.png"):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)
 
video = cv2.VideoWriter("faces_video.mp4", 0, 1, (width, height)) 