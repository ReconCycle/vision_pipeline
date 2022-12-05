import numpy as np
from scipy import spatial
import cv2
from rich import print
from probreg import cpd
from probreg import callbacks
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from helpers import scale_img
from shapely.geometry import Polygon
import matplotlib.pyplot as plt


class WorkSurfaceDetection:
    def __init__(self, img, border_width=0, debug=False):
        
        self.border_width = border_width
        self.debug = debug
        
        self.img_width = None
        self.img_height = None
        self.coord_transform = None
        self.coord_transform_inv = None
        self.circles = None
        self.circles_ignoring = None
        
        self.font_face = cv2.FONT_HERSHEY_DUPLEX
        self.font_scale = 1.0
        self.font_thickness = 1
        
        # affine transform
        self.pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
        self.unpad = lambda x: x[:, :-1]
        
        if isinstance(img, str):
            img = np.array(cv2.imread(img))
        else:
            img = np.array(img)
        
        if img is not None:
            print("running worksurface detection...")
            self.run_detection(img)
        else:
            print("No image found!")

    @staticmethod
    def generate_all_sides(m_bottom_dict, all_m_dict):
        top = lambda x, y: [0.6 - x, 0.6 - y]
        right = lambda x, y: [0.6 - y, x]
        bottom = lambda x, y: [x, y]
        left = lambda x, y: [y, 0.6 - x]
    
        for key in m_bottom_dict:
            for translate, name in [(top, "t"), (right, "r"), (bottom, "b"), (left, "l")]:
                new_key = key + "_" + name
                all_m_dict[new_key] = translate(*m_bottom_dict[key])

    @staticmethod
    def flip_horizontal_m_dict(m_dict):
        copy_dict = m_dict.copy()
        for key, val, in copy_dict.items():
            val[1] = 0.60 - val[1]
        return copy_dict

    def run_detection(self, img):
        
        self.img_width = img.shape[1]
        self.img_height = img.shape[0]
        
        self.corners_px_dict = {}
        self.bolts_px_dict = {}
        
        self.corners_m_dict = {}
        self.bolts_m_dict = {}

        # this will be generated:        
        # {
        #     'corner_t': (0.6, 0.6),
        #     'corner_r': (0.6, 0),
        #     'corner_b': (0, 0),
        #     'corner_l': (0, 0.6)
        # }
        
        # bolts and corners in real world coords
        self.corner_m_bottom_dict = {
            'corner': [0, 0]
        }
        self.bolts_m_bottom_dict = {
            'bolt0': [0.035, 0.035],
            'bolt1': [0.095, 0.015],
            'bolt2': [0.095, 0.045],
            'bolt3': [0.160, 0.030],
            'bolt4': [0.230, 0.030],
            'calibrationmount': [0.03, 0.3],
            'bolt5': [0.60 - 0.230, 0.030],
            'bolt6': [0.60 - 0.160, 0.030],
            'bolt7': [0.60 - 0.095, 0.045],
            'bolt8': [0.60 - 0.095, 0.015],
            # 'bolt9': [0.60 - 0.035, 0.035], # the last bolt gets replicated as we generate all the sides
        }
        
        # we rotate in steps of 90 degrees to now get all the bolt hole measurements
        self.generate_all_sides(self.bolts_m_bottom_dict, self.bolts_m_dict)
        self.generate_all_sides(self.corner_m_bottom_dict, self.corners_m_dict)
        
        print("corners_m_dict", self.corners_m_dict)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 1st estimate for affine transform using only bolts
        self.find_bolts()
        self.bolt_point_matching()
        self.compute_affine_transform()
        
        self.estimate_corners_using_transform()
        
        # 2nd estimate for affine transformation using also corners
        #! this usually makes things worse
        # self.improve_corner_estimate_using_corner_detection(img)
        # self.compute_affine_transform()
        
        # for debugging, draw everything
        # if self.debug:
        #     self.draw_corners_and_circles(img, show=True)
        
    def bolt_point_matching(self):
        source = self.circles[:, :2].astype(np.float32) # x, y pairs
        
        target_keys = list(self.bolts_m_dict.keys())
        target = np.array(list(self.bolts_m_dict.values())).astype(np.float32)
        
        # flip the bolts to go from world coords to opencv coords
        target_flipped = target.copy()
        target_flipped[:, 1] = 0.6 - target_flipped[:, 1]
        
        # scale pixels to the ballpark range of the work surface in meters
        source_scaled = (source / np.max(source))

        # center the pixels to that of the work surface
        source_centroid = source_scaled.mean(axis=0)
        target_centroid = target_flipped.mean(axis=0)
        diff = target_centroid - source_centroid
        source_scaled += diff
        
        if self.debug:
            print("source_centroid", source_centroid)
            print("target_centroid", target_centroid)
            
            print("sourcsource_scalede.shape", source_scaled.shape)
            print("real_world_pts", target_flipped.shape)

            print("source_scaled", source_scaled.shape)
            print("target", target_flipped.shape)

        cbs = [callbacks.Plot2DCallback(source_scaled, target_flipped)]
        
        # a larger w will cause the point cloud model to approach a uniform 
        res = cpd.registration_cpd(source_scaled, target_flipped, 'affine', w=0.5, maxiter=200, tol=0.00015, callbacks=cbs) # affine vs nonrigid
        
        if self.debug:
            print("res.q", res.q)
            print("sigma2", res.sigma2)
            affine_m = np.array(res.transformation.b).T
            affine_t = np.array([res.transformation.t]).T
            affine = np.hstack((affine_m, affine_t))
            affine = np.vstack((affine, np.array([0, 0, 1])))
            
            print("affine matrix", affine_m)
            print("affine translation", affine_t)
            print("affine", affine)

            plt.show()
        
        result = np.copy(source_scaled)
        result = res.transformation.transform(result)
        # the same as:
        # result = np.dot(result, res.transformation.b.T) + res.transformation.t
        
        # print("result", result.shape)
        # print("target", target.shape)

        #! I think I should use nearest neighbour instead!
        #! linear_sum_assignment optimises also for completely wrong pairs, instead of ignoring them

        C = cdist(target_flipped, result)

        _, matching_idxs = linear_sum_assignment(C)
        
        result_matching = result[matching_idxs]
        matching_dists = np.linalg.norm(target_flipped - result_matching, axis=1)
        
        print("matching_idxs", matching_idxs, matching_idxs.shape)
        
        # todo: remove matchings with too large distances
        for i in np.arange(len(matching_idxs)):
            if matching_dists[i] > 0.05: # value in meters
                matching_idxs[i] = -1
                
                
        max_error = np.linalg.norm(target_flipped - result_matching, axis=1).max()
        mean_error = np.linalg.norm(target_flipped - result_matching, axis=1).mean()
        
        if self.debug:
            plt.plot(target_flipped[:,0], target_flipped[:,1],'bo', markersize = 10)
            plt.plot(result[:,0], result[:,1],'rs',  markersize = 7)
            for p in range(target_flipped.shape[0]):
                if matching_idxs[p] > -1:
                    plt.plot([target_flipped[p,0], result[matching_idxs[p],0]], [target_flipped[p,1], result[matching_idxs[p],1]], 'k')
            plt.show()

        print("max_error", max_error)
        print("mean_error", mean_error)
        
        #! Check that the matching didn't rotate all the points around! 
        #! Is the top left point in the top-left quadrant of the image?
        
        # add bolts in pixels to dict
        for i, matching_idx in enumerate(matching_idxs):
            if matching_idx > -1:
                key = target_keys[i]
                self.bolts_px_dict[key] = source[matching_idx]

    
    def find_bolts(self):

        # Finds circles in a grayscale image using the Hough transform
        # minDist: minimum distance between detected circles
        # param1: detects strong edges, as pixels which have gradient value higher than param1
        # param 2: it is the accumulator threshold for the circle centers at the detection stage
        #          The smaller it is, the more false circles may be detected.
        circles = cv2.HoughCircles(self.blur, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                                    param1=50, param2=20, minRadius=10, maxRadius=20)

        if circles is not None:
            circles = np.array([[x, y, r] for (x, y, r) in circles[0]])
            
            # remove circles that are on the edge of the image, these can lead to incorrect results
            # in the case that worksurfaces are next to each other
            circles_ignoring = []
            circles_inner_region = []
            for (x, y, r) in circles:
                if x < self.border_width \
                    or x > self.img_width - self.border_width \
                    or y < self.border_width \
                    or y > self.img_height - self.border_width:
                    
                    # circle is close to edge of image, so ignore
                    # for visualisation only
                    circles_ignoring.append([x, y, r])
                else:
                    circles_inner_region.append([x, y, r])
            
            circles = np.array(circles_inner_region)
            circles_ignoring = np.array(circles_ignoring)
                
            self.circles = circles
            self.circles_ignoring = circles_ignoring
            
    
    def improve_corner_estimate_using_corner_detection(self, img):
        
        """in the neighbourhood of the corner estimation, use Harris corner
        detection to find a better approximation for the corner
        """
        img_show = np.zeros(img.shape[:3], np.uint8)
        # iterate over corners
        for key, value in self.corners_px_dict.items():
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
                self.corners_px_dict[key] = corners[closest_index]
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
                
                img_show += img_masked
                
        
        if self.debug:
            cv2.imshow("corners", scale_img(img_show))
            cv2.waitKey()
            cv2.destroyAllWindows()

    def compute_affine_transform(self):

        # create arrays for affine transform
        points_px = []
        points_m = []
        # append corners
        for key, value in self.corners_px_dict.items():
            if value is not None and key in self.corners_m_dict and self.corners_m_dict[key] is not None:
                points_px.append(value)
                points_m.append(self.corners_m_dict[key])
        
        # append keys
        for key, value in self.bolts_px_dict.items():
            if value is not None and key in self.bolts_m_dict and self.bolts_m_dict[key] is not None:
                points_px.append(value)
                points_m.append(self.bolts_m_dict[key])

        points_px = np.array(points_px)
        points_m = np.array(points_m)

        X = self.pad(points_px)
        Y = self.pad(points_m)

        # Solve the least squares problem X * A = Y
        # to find our transformation matrix A
        A, res, rank, s = np.linalg.lstsq(X, Y, rcond=None)
        self.coord_transform = lambda x: self.unpad(np.dot(self.pad(x), A))
        
        A_inv = np.linalg.solve(A.T.dot(A), A.T)
        self.coord_transform_inv = lambda x: self.unpad(np.dot(self.pad(x), A_inv))
        
        # max_error = np.abs(points_m - self.pixels_to_meters(points_px)).max()
        max_error = np.linalg.norm(points_m - self.pixels_to_meters(points_px), axis=1).max()
        mean_error = np.linalg.norm(points_m - self.pixels_to_meters(points_px), axis=1).mean()
        
        if max_error > 0.02:
            print("[red]Max error in work surface position is: " + str(max_error) +"[/red]")
        
        if self.debug:
            # print("points_px", points_px)
            # print("points_m", points_m)
            
            # print("Target:", points_m)
            # print("Result:", self.pixels_to_meters(points_px))
            print("max_error", max_error)
            print("mean_error", mean_error)

        print("self.meters_to_pixels", self.meters_to_pixels(np.array([0.0, 0.0])))
        print("and back...", self.pixels_to_meters(self.meters_to_pixels(np.array([0.0, 0.0]))))
        
        
    def estimate_corners_using_transform(self):
        for key, value in self.corners_m_dict.items():
            corner_in_meters = self.meters_to_pixels(np.array(value))
            self.corners_px_dict[key] = corner_in_meters
            if corner_in_meters[0] > self.img_width or corner_in_meters[0] < 0 \
                or corner_in_meters[1] > self.img_height or corner_in_meters[1] < 0:
                print("[red]Corner estimate is out of bounds! " + str(corner_in_meters[0]) + ", " + str(corner_in_meters[1]) + "[/red]")
        
        if self.debug:
            print("self.corners_px_dict", self.corners_px_dict)
        
    def draw_corners_and_circles(self, img, show=False):
        # draw stuff on image
        if self.circles is not None:
            # draw all detections in green
            for (x, y, r) in self.circles:
                # Draw the circle in the output image
                cv2.circle(img, (int(x), int(y)), int(r), (255,0,0), 3)
                # Draw a cross in the output image
                cv2.drawMarker(img, (int(x), int(y)), color=[255,0,0],
                    markerType=cv2.MARKER_CROSS,
                    markerSize=20,
                    thickness=2,
                    line_type=8)
                
            for (x, y, r) in self.circles_ignoring:
                # Draw the circle in the output image
                cv2.circle(img, (int(x), int(y)), int(r), (0,0,255), 3)
                # Draw a cross in the output image
                cv2.drawMarker(img, (int(x), int(y)), color=[0,0,255],
                    markerType=cv2.MARKER_CROSS,
                    markerSize=20,
                    thickness=2,
                    line_type=8)
        
            # for bolts and calibration mounts
            for key in self.bolts_px_dict:
                if self.bolts_px_dict[key] is not None:
                    if len(self.bolts_px_dict[key]) == 3:
                        x, y, r = self.bolts_px_dict[key]
                    else:
                        x, y = self.bolts_px_dict[key]
                        r = None
                    if r is not None:
                        cv2.circle(img, (int(x), int(y)), int(r), (0,255,0), 3)
                    # Draw a cross in the output image
                    cv2.drawMarker(img, (int(x), int(y)), color=[0,255,0],
                        markerType=cv2.MARKER_CROSS,
                        markerSize=20,
                        thickness=2,
                        line_type=8)
                    
                    cv2.putText(img, key, (int(x)-100, int(y)-20), self.font_face, self.font_scale, [0,255,0], self.font_thickness, cv2.LINE_AA)
                    
        if show:
            cv2.imshow("1", scale_img(img))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

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
    