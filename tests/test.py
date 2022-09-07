import cv2
import numpy as np
#import matplotlib

x = np.zeros((480, 640, 3))
x[400, 100, :] = (255, 255, 255)
cv2.circle(x, [400, 100], 5, (0, 255, 0), -1)

cv2.imshow('asdf', x)

cv2.waitKey()