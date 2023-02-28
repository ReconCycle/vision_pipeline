import sys
import os
import cv2
import numpy as np
from rich import print

import helpers

class SimpleDetector:
    def __init__(self) -> None:
        pass
    
    def run(self, img, visualise=False):
        img_copy = np.copy(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # define a mask
        mask_top = 200
        mask_left = 200
        mask_bottom = 200
        mask_right = 200
        
        # crop image
        gray = gray[mask_left:(img.shape[0] - mask_right), mask_top: (img.shape[1] - mask_bottom)]
        # gray = cv2.GaussianBlur(gray,(3,3),cv2.BORDER_DEFAULT)
        
        # ret, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        # https://stackoverflow.com/questions/61432335/blob-detection-in-python
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 7)
        
        # otsu_threshold, thresh = cv2.threshold(
        #     gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        # )

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        blob = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31,31))
        blob2 = cv2.morphologyEx(blob, cv2.MORPH_CLOSE, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        blob3 = cv2.morphologyEx(blob2, cv2.MORPH_OPEN, kernel)

        # blur = cv2.GaussianBlur(thresh, (7, 7), 0)
        # blur = cv2.medianBlur(blur, 3)   #to remove salt and paper noise
        # cv2.floodFill(blur, np.zeros((img.shape[0]+2, img.shape[1]+2), np.uint8), (0, 0), 0)
        
        # Remove some small noise if any.
        # blur = cv2.GaussianBlur(thresh,(25,25),cv2.BORDER_DEFAULT)
        # dilate = cv2.dilate(blur,None)
        # erode = cv2.erode(dilate,None)
        # kernel = np.ones((2,2),np.uint8)
        # morph = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, kernel)
        #to strength week pixels
        # dilate = cv2.dilate(morph,kernel,iterations = 5)

        if visualise:
            cv2.imshow('gray', helpers.scale_img(gray))
            cv2.imshow('thresh', helpers.scale_img(thresh))
            cv2.imshow('blob', helpers.scale_img(blob))
            cv2.imshow('blob2', helpers.scale_img(blob2))
            cv2.imshow('blob3', helpers.scale_img(blob3))
            # cv2.imshow('dilate', helpers.scale_img(dilate))
            # cv2.imshow('erode', helpers.scale_img(erode))

        # Find contours with cv2.RETR_CCOMP
        contours, hierarchy = cv2.findContours(blob3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        good_cnts = []
        good_boxes = []

        for i,cnt in enumerate(contours):
            # Check if it is an external contour and its area is more than 100
            cnt = cnt + [mask_left, mask_top]
            cnt = cv2.convexHull(cnt)
            # print("cv2.contourArea(cnt)", i, ": ", cv2.contourArea(cnt))
                
            # TODO: magic number contourArea
            # hierarchy[0,i,3] == -1
            if cv2.contourArea(cnt) > 3000 and cv2.contourArea(cnt) < 30000:
                # print("cnt", cnt.shape)
                
                cv2.drawContours(img_copy, [cnt], -1, (0, 255, 0), 3)
                
                good_cnts.append(cnt)
                
                x,y,w,h = cv2.boundingRect(cnt)
                
                # format: tlbr
                good_boxes.append(np.array([x,y,x+w,y+h]))
                cv2.rectangle(img_copy,(x,y),(x+w,y+h),(0,255,0),2)

                m = cv2.moments(cnt)
                cx,cy = m['m10']/m['m00'],m['m01']/m['m00']
                cv2.circle(img_copy,(int(cx),int(cy)),3,255,-1)
            else:
                cv2.drawContours(img_copy, [cnt], -1, (0, 0, 255), 3)

        if visualise:
            cv2.imshow('img', helpers.scale_img(img_copy))
            # cv2.imwrite('sofsqure.png',img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return good_cnts, good_boxes
        
if __name__ == '__main__':
    simple_detector = SimpleDetector()
    
    img = cv2.imread("experiments/datasets/2023-02-20_hca_backs/hca_11/0030.jpg")
    # img = cv2.imread("experiments/datasets/2023-02-20_hca_backs/hca_2/0012.jpg")
    # img = cv2.imread("experiments/datasets/2023-02-20_hca_backs/hca_0/0001.jpg")
    # img = cv2.imread("experiments/datasets/2023-02-20_hca_backs/hca_3/0001.jpg")
    # img = cv2.imread("experiments/datasets/2023-02-20_hca_backs/hca_1/0017.jpg")
    simple_detector.run(img, visualise=True)
    