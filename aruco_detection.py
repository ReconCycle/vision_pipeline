import cv2

class ArucoDetection:
    def __init__(self):
        pass
    
    def run(self, img, worksurface_detection=None):
        draw_img = img.copy()
        
        arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
        arucoParams = cv2.aruco.DetectorParameters_create()
    
        corners, ids, rejected = cv2.aruco.detectMarkers(img, arucoDict, parameters=arucoParams)
        
        cv2.aruco.drawDetectedMarkers(draw_img, corners)
        
        print("aruco ids detected:", ids)
        
        for (markerCorner, markerID) in zip(corners, ids):
            
            topLeft, topRight, bottomRight, bottomLeft = self.extract_corners(markerCorner)

            cv2.putText(draw_img, str(markerID), (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return draw_img
    
    @staticmethod
    def extract_corners(corner_matrix):
        # extract the marker corners (which are always returned in top-left, top-right, bottom-right, and bottom-left order)
        topLeft, topRight, bottomRight, bottomLeft = corner_matrix.reshape((4, 2))
        
        # convert each of the (x, y)-coordinate pairs to integers
        topRight = (int(topRight[0]), int(topRight[1]))
        bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
        bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
        topLeft = (int(topLeft[0]), int(topLeft[1]))

        return topLeft, topRight, bottomRight, bottomLeft