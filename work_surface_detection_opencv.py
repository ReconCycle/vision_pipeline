import numpy as np
from scipy import spatial
import cv2
from rich import print

from helpers import scale_img
from shapely.geometry import Polygon


def get_corner_bolts(pts):
    """order points in: top-left, top-right, bottom-right, and bottom-left
    get the pts that are the most in these corners are return these
    
    Source: https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
    
    Args:
        pts (arr[*, 3]): array of points, where each point is [x, y, r]
        or:
        pts (arr[*, 2]): array of points, where each point is [x, y]

    Returns:
        arr[4,2]: corners
    """
    points_dim = pts.shape[1]
    
    print("pts.shape", pts.shape)
    
    corners = np.zeros((4, points_dim), dtype = "float32")
    
    # determine if pts are [x, y, r] or [x, y]
    summing_pts = pts
    if points_dim == 3:
        summing_pts = pts[:, :2]
        
    s = np.sum(summing_pts, axis = 1)
    
    print("sum", s)
    
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    corners[0] = pts[np.argmin(s)]
    corners[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(summing_pts, axis = 1)
    corners[1] = pts[np.argmin(diff)]
    corners[3] = pts[np.argmax(diff)]
    
    return corners


class WorkSurfaceDetection:
    def __init__(self, img, debug=False):
        
        self.debug = debug
        
        self.img_width = None
        self.img_height = None
        self.coord_transform = None
        self.coord_transform_inv = None
        self.circles = None
        self.points_px_dict = None
        self.points_m_dict = None
        
        self.font_face = cv2.FONT_HERSHEY_DUPLEX
        self.font_scale = 1.0
        self.font_thickness = 1
        
        if isinstance(img, str):
            img = np.array(cv2.imread(img))
        else:
            img = np.array(img)
            
        if img is not None:
            self.run_detection(img)
        else:
            print("No image found!")


    def run_detection(self, img):
        
        self.img_width = img.shape[1]
        self.img_height = img.shape[0]

        self.points_px_dict = {
            'corner0': None,
            'corner1': None,
            'corner2': None,
            'corner3': None,
            'bolt0': None,
            'bolt1': None,
            'bolt2': None,
            'bolt3': None,
            'calibrationmount0': None,
            'calibrationmount1': None,
            'calibrationmount2': None,
            'calibrationmount3': None,
        }
                
        self.points_m_dict = {
            'corner0': [0, 0.6],                    # top-left
            'corner1': [0.6, 0.6],                  # top-right
            'corner2': [0.6, 0],                    # bottom-right
            'corner3': [0, 0],                      # bottom-left
            
            # bolts are 0.035m in from the edge
            'bolt0': [0.035, 0.6 - 0.035],          # top-left
            'bolt1': [0.6 - 0.035, 0.6 - 0.035],    # top-right
            'bolt2': [0.6 - 0.035, 0.035],          # bottom-right
            'bolt3': [0.035, 0.035],                # bottom-left
            
            # calibration mounts are 0.03m in from the edge
            'calibrationmount0': [0.3, 0.6 - 0.03], # top-center
            'calibrationmount1': [0.6 - 0.03, 0.3], # right-center
            'calibrationmount2': [0.3, 0.03],       # bottom-center
            'calibrationmount3': [0.03, 0.3],       # left-center
        }

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 1st estimate for affine transform using only bolts + calibration mount
        self.estimate_bolts_and_calibration_mounts()
        self.compute_affine_transform()
        self.estimate_corners_using_transform()
        
        # 2nd estimate for affine transformation using also corners
        self.improve_corner_estimate_using_corner_detection(img)
        self.compute_affine_transform()
        
        # for debugging, draw everything
        self.draw_corners_and_circles(img)
        
    def estimate_bolts_and_calibration_mounts(self):

        # Finds circles in a grayscale image using the Hough transform
        # minDist: minimum distance between detected circles
        # param1: detects strong edges, as pixels which have gradient value higher than param1
        # param 2: it is the accumulator threshold for the circle centers at the detection stage
        #          The smaller it is, the more false circles may be detected.
        circles = cv2.HoughCircles(self.blur, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
                                    param1=50, param2=20, minRadius=10, maxRadius=20)

        bolts = None

        if circles is not None:
            circles = np.array([[x, y, r] for (x, y, r) in circles[0]])
            
            # remove circles that are on the edge of the image, these can lead to incorrect results
            # in the case that worksurfaces are next to each other
            circles_inner_region = []
            border_width = 30
            for (x, y, r) in circles:
                if x < border_width \
                    or x > self.img_width - border_width \
                    or y < border_width \
                    or y > self.img_height - border_width:
                    
                    # circle is close to edge of image, so ignore
                    pass
                else:
                    circles_inner_region.append([x, y, r])
            
            circles = np.array(circles_inner_region)

            # find the corner bolts
            bolts = get_corner_bolts(circles)
            
            # write bolts to dictionary
            for i, bolt in enumerate(bolts):
                self.points_px_dict["bolt"+ str(i)] = bolt

            # compute which circle corresponds to calibration mount, if any
            # and write calibration mount(s) to dictionary
            midpoints = []
            midpoints.append((bolts[0] + bolts[1])/2)
            midpoints.append((bolts[1] + bolts[2])/2)
            midpoints.append((bolts[2] + bolts[3])/2)
            midpoints.append((bolts[3] + bolts[0])/2)
            
            # the hough circle for the calibration mount should be in the neighbourbood of 30px
            # from our midpoint estimation
            # the calibration mount is not exactly at the midpoint
            epsilon = 30
            for [x, y, r] in circles:
                for i, midpoint in enumerate(midpoints):
                    if np.linalg.norm(np.array([x, y])-midpoint[:2]) < epsilon:
                        self.points_px_dict["calibrationmount" + str(i)] = [x, y, r]
       
            # if self.debug:
            #     print("circles", circles)
            #     print("bolts", bolts)
            #     print("midpoints", midpoints)
                
        self.circles = circles
            
    def improve_corner_estimate_using_corner_detection(self, img):
        
        """in the neighbourhood of the corner estimation, use Harris corner
        detection to find a better approximation for the corner
        """

        for key, value in self.points_px_dict.items():
            # iterate over corners
            if value is not None and key.startswith("corner"):
                x, y = value
                
                # create a circle mask where we apply harris corner detection in
                circle_radius = 100
                corner_mask = np.zeros(img.shape[:2], np.uint8)
                cv2.circle(corner_mask,(int(x), int(y)), circle_radius, 255, thickness=-1)
                img_masked = cv2.bitwise_and(img, img, mask=corner_mask) # for visualisation
                blur_masked = cv2.bitwise_and(self.blur, self.blur, mask=corner_mask)
                
                # get rid of noisier corners from background by using erode + dilate
                kernel = np.ones((5,5), np.uint8)
                blur_masked = cv2.erode(blur_masked, kernel, iterations=1)
                blur_masked = cv2.dilate(blur_masked, kernel, iterations=1)
                # cv2.imshow(key + "_blur+dilate", scale_img(blur_masked))
                
                # if self.debug:
                #     cv2.imshow(key + "blur", scale_img(blur_masked))
                
                # use Harris corner detection
                # blockSize - It is the size of neighbourhood considered for corner detection
                # ksize - Aperture parameter of Sobel derivative used. As size increases, edges will get more blurry
                # k - Harris detector free parameter in the equation. Bigger k, you will get less false corners. Smaller k you will get a lot more corners. Constant in the range: [0.04,0.06]
                dst = cv2.cornerHarris(blur_masked, blockSize=20, ksize=5, k=0.04)
                
                # get better accuracy: https://docs.opencv.org/3.4/dc/d0d/tutorial_py_features_harris.html
                dst = cv2.dilate(dst,None)
                ret, dst = cv2.threshold(dst,0.001*dst.max(),255,0) # was 0.01 but changed to 0.001 to see more corners
                dst = np.uint8(dst)
                # find centroids
                ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
                # define the criteria to stop and refine the corners
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
                corners = cv2.cornerSubPix(blur_masked,np.float32(centroids),(5,5),(-1,-1),criteria)
                
                # now find the corner closest to the original estimation
                tree = spatial.KDTree(corners)
                closest_dist, closest_index = tree.query([(x, y)])
                closest_dist = closest_dist[0]
                closest_index = closest_index[0]
                
                # if the precise corner is close enough to the estimate then we assume there were no errors
                # the new position has to be within 20px of the estimate
                if closest_dist < 20:
                    # update corner values
                    self.points_px_dict[key] = corners[closest_index]
                else:
                    print("[red]Corner detection too far off estimate! [/red]")
                    print("[red] Using original estimate from bolt position for: " + key + "[/red]")
                
                # Now draw them
                if self.debug:
                    print("img.shape", img.shape)
                    print("x, y", x, y)
                    print("harris corners shape", corners.shape)
                    # print("harris corners", corners)
                    
                    print("closest_corner: dist, index", closest_dist, closest_index)
                    print("closest_corner", corners[closest_index])
                    print("")
                    
                    res = np.hstack((centroids,corners))
                    res = np.int0(res)
                    
                    # Threshold for an optimal value, it may vary depending on the image.
                    img_masked[dst>0.01*dst.max()]=[0,0,255]
                    
                    img_masked[res[:,1],res[:,0]]=[0,0,255] # red centroids
                    img_masked[res[:,3],res[:,2]] = [0,255,0] # green corners
                
                    cv2.drawMarker(img_masked, tuple(corners[closest_index].astype(int)), color=[0,255,0],
                        markerType=cv2.MARKER_CROSS,
                        markerSize=20,
                        thickness=1,
                        line_type=8)
                    
                    cv2.imshow(key, scale_img(img_masked))

    def compute_affine_transform(self):

        # create arrays for affine transform
        points_px_arr = []
        points_m_arr = []
        for key, value in self.points_px_dict.items():
            if value is not None and key in self.points_m_dict and self.points_m_dict[key] is not None:
                points_px_arr.append(value[:2]) # only [x, y] not radius
                points_m_arr.append(self.points_m_dict[key])

        points_px_arr = np.array(points_px_arr)
        points_m_arr = np.array(points_m_arr)

        self.pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
        self.unpad = lambda x: x[:, :-1]

        X = self.pad(points_px_arr)
        Y = self.pad(points_m_arr)

        # Solve the least squares problem X * A = Y
        # to find our transformation matrix A
        A, res, rank, s = np.linalg.lstsq(X, Y, rcond=None)
        self.coord_transform = lambda x: self.unpad(np.dot(self.pad(x), A))
        
        A_inv = np.linalg.solve(A.T.dot(A), A.T)
        self.coord_transform_inv = lambda x: self.unpad(np.dot(self.pad(x), A_inv))
        
        max_error = np.abs(points_m_arr - self.pixels_to_meters(points_px_arr)).max()
        
        if max_error > 0.02:
            print("[red]Max error in work surface position is: " + str(max_error) +"[/red]")
        
        if self.debug:
            print("points_px_arr", points_px_arr)
            print("points_m_arr", points_m_arr)
            
            print("Target:", points_m_arr)
            print("Result:", self.pixels_to_meters(points_px_arr))
            print("Max error:", max_error)
            
        

        print("self.meters_to_pixels", self.meters_to_pixels(np.array([0.0, 0.0])))
        
        
    def estimate_corners_using_transform(self):
        for key, value in self.points_px_dict.items():
            if value is None and key.startswith("corner"):
                corner_in_meters = self.meters_to_pixels(np.array(self.points_m_dict[key]))
                self.points_px_dict[key] = corner_in_meters
                if corner_in_meters[0] > self.img_width or corner_in_meters[0] < 0 \
                    or corner_in_meters[1] > self.img_height or corner_in_meters[1] < 0:
                    print("[red]Corner estimate is out of bounds! " + str(corner_in_meters[0]) + ", " + str(corner_in_meters[1]) + "[/red]")
        
        if self.debug:
            print("self.points_px_dict", self.points_px_dict)
        
    def draw_corners_and_circles(self, img):
        # draw stuff on image
        if self.circles is not None:
            # draw all detections in green
            for (x, y, r) in self.circles:
                # Draw the circle in the output image
                cv2.circle(img, (int(x), int(y)), int(r), (0,255,0), 3)
                # Draw a cross in the output image
                cv2.drawMarker(img, (int(x), int(y)), color=[0,255,0],
                    markerType=cv2.MARKER_CROSS,
                    markerSize=20,
                    thickness=2,
                    line_type=8)
        
            # for bolts and calibration mounts
            for key in self.points_px_dict:
                if self.points_px_dict[key] is not None:
                    if len(self.points_px_dict[key]) == 3:
                        x, y, r = self.points_px_dict[key]
                    else:
                        x, y = self.points_px_dict[key]
                        r = None
                    if r is not None:
                        cv2.circle(img, (int(x), int(y)), int(r), (0,0,255), 3)
                    # Draw a cross in the output image
                    cv2.drawMarker(img, (int(x), int(y)), color=[0,0,255],
                        markerType=cv2.MARKER_CROSS,
                        markerSize=20,
                        thickness=2,
                        line_type=8)
                    
                    cv2.putText(img, key, (int(x)-100, int(y)), self.font_face, self.font_scale, [0,0,255], self.font_thickness, cv2.LINE_AA)

            # cv2.imshow("0", scale_img(blur))
            # cv2.imwrite('./example_work_sufrace_detection_opencv.jpg', img)
        if self.debug:
            cv2.imshow("1", scale_img(img))
            cv2.waitKey(0)

    def pixels_to_meters(self, coords, depth=None):
        if isinstance(coords, Polygon):
            # coords.exterior.coords[:-1] for non repeated list
            polygon_px_coords = np.asarray(list(coords.exterior.coords))
            polygon_coords = self.coord_transform(polygon_px_coords)
            if depth is not None:
                polygon_coords = np.pad(polygon_coords, [(0, 0), (0, 1)], mode='constant', constant_values=depth)
            
            return Polygon(polygon_coords)
            
        elif isinstance(coords, tuple) or len(coords.shape) == 1:
            # single coordinate pair.
            # todo: add depth option
            return self.coord_transform(np.array([coords]))[0]
        else:
            # assume array of coordinate pairs.
            # Each row contains a coordinate (x, y) pair
            # todo: add depth option
            return self.coord_transform(coords)
        
    def meters_to_pixels(self, coords):
        # todo: deal with depth
        if isinstance(coords, Polygon):
            polygon_coords = np.asarray(list(coords.exterior.coords))
            polygon_px_coords = self.coord_transform_inv(polygon_coords)
            return Polygon(polygon_px_coords)
        elif isinstance(coords, tuple) or len(coords.shape) == 1:
            # single coordinate pair.
            return self.coord_transform_inv(np.array([coords]))[0]
        else:
            # assume array of coordinate pairs.
            # Each row contains a coordinate (x, y) pair
            return self.coord_transform_inv(coords)
        
if __name__ == '__main__':
    img = cv2.imread("data_full/dlc/dlc_work_surface_jsi_05-07-2021/labeled-data/raw_work_surface_jsi_08-07-2021/img000.png")
    work_surface_det2 = WorkSurfaceDetection(img, debug=True)
    