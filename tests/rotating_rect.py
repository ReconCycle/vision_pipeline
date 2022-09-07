import cv2
import numpy as np
from scipy.spatial.transform import Rotation

def main():
    angle = 0

    font_face = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.5
    font_thickness = 1

    while cv2.waitKey() != 27:
        image = np.zeros((640, 1000, 3))

        rect1 = ((200, 300), (150, 300), angle) # center, width, height, angle
        box = np.int0(cv2.boxPoints(rect1)) #Convert into integer values
        cv2.drawContours(image,[box],0, (0,0,255) , 2)

        for i in np.arange(len(box)):
            cv2.putText(image, str(i)+": (" +str(box[i][0]) + "," + str(box[i][1])+")",
                            tuple(box[i]), font_face, font_scale, [255, 255, 255], font_thickness, cv2.LINE_AA)

        # print("box", box)

        # move box to the right
        shift = (500, 0)
        box_shifted = box + np.array([shift, shift, shift, shift])

        rect2 = cv2.minAreaRect(box_shifted)
        # rect2 = (center, size, angle) rotatedrect

        box2 = np.int0(cv2.boxPoints(rect2))
        center = np.int0(rect2[0])
        # width, height = rect2[0] # ! this one doesn't swap width and height!!
        rot = rect2[2]
        changing_width, changing_height = box_w_h(box2)

        cv2.drawContours(image,[box2],0,(0,0,255),2)
        for i in np.arange(len(box2)):
            cv2.putText(image, str(i)+": (" +str(box2[i][0]) + "," + str(box2[i][1])+")",
                            tuple(box2[i]), font_face, font_scale, [255, 255, 255], font_thickness, cv2.LINE_AA)
        
        

        if changing_height > changing_width:
            correct_angle = rot
        else:
            correct_angle = rot - 90

        correct_quat = Rotation.from_euler('z', correct_angle, degrees=True).as_quat()

        correct_height = changing_height
        correct_width = changing_width
        if changing_height < changing_width:
            correct_height = changing_width
            correct_width = changing_height

        image = rotated_line(tuple((100, 100)), 0, 50, image) # 0 degrees
        image = rotated_line(tuple((100, 100)), 45, 50, image) # 45 degrees

        image = rotated_line(center, correct_angle, 60, image, colour=(0, 255, 0))
        image = rotated_line(center, rot, 50, image)

        cv2.imshow("rectangles", image)

        # Print the angle values.
        print("---\n")
        print("Original angle:", angle % 360)
        print("rot deg:", rot)
        print("correct_angle", correct_angle)
        print("correct quat:", correct_quat)
        print("")
        print("correct_width:", correct_width)
        print("correct_height", correct_height)
        print("")
        print("changing_width:", changing_width)
        print("changing_height", changing_height)
        print("")
        print("---\n")

        angle += 10

def rotated_line(point, angle, length, image, colour=(0, 0, 255)):
    angle_rad = np.deg2rad(angle)
    x2 = point[0] + length * np.cos(angle_rad)
    y2 = point[1] + length * np.sin(angle_rad)
    point2 = tuple([int(x2), int(y2)])

    return cv2.arrowedLine(image, tuple(point), point2, colour, 3, tipLength = 0.5)

def box_w_h(box):
    return np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])

def rotate(points, angle):
    ANGLE = np.deg2rad(angle)
    c_x, c_y = np.mean(points, axis=0)
    return np.array(
        [
            [
                c_x + np.cos(ANGLE) * (px - c_x) - np.sin(ANGLE) * (py - c_x),
                c_y + np.sin(ANGLE) * (px - c_y) + np.cos(ANGLE) * (py - c_y)
            ]
            for px, py in points
        ]
    ).astype(int)


if __name__ == '__main__':
    main()

