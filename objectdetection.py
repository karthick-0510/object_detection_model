# Import libraries
import cv2
import numpy as np
import cvlib as cv
from PIL import Image
from cvlib.object_detection import draw_bbox
import matplotlib.pyplot as plt
%matplotlib inline

# Read images
img = Image.open('car.jpg')
img = np.array(img)

# Detect and drawbox around objects
bbox, label, conf = cv.detect_common_objects(img, model='yolov3')
output_image = draw_bbox(img, bbox, label, conf)

# Plot the image with  predictions
plt.imshow(output_image)
