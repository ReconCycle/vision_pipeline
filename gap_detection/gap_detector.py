
# Numpy and scikit-learn
import numpy as np
from skimage.filters import threshold_otsu, threshold_minimum, threshold_li
from skimage.filters import threshold_yen
from scipy.spatial import ConvexHull
import sklearn.cluster as cluster
import hdbscan
import time
# import config

# Own Modules
import gap_detection.helpers as helpers

# Events? Dont know what threding is for
import threading

class GapDetector:
    def __init__(self):
        self.detected_gap = threading.Event()  # event variable to signal gap detection
        self.image_received = threading.Event()  # event variable to signal gap detection
         
        self.potential_gaps_pub = []
        self.convex_hull_marker_pub = []
        self.centers_marker_pub = []
        self.volume_text_marker_pub = []

        self.depth_axis = 2
        self.clustering_mode = 3 # 0 to 3
        self.create_evaluation = False
        self.automatic_thresholding = -1 # -1 to 3
        self.min_gap_volume = 0.1 # 0.0 to 200.0
        self.max_gap_volume = 20.0 # 0.0 to 200.0
        self.KM_number_of_clusters = 3 # 1 to 10
        self.B_branching_factor = 50 # 2 to 200
        self.B_threshold = 0.015 # 0.0 to 1.0
        self.DB_eps = 0.007 # 0.0001 to 0.02
        self.HDB_min_cluster_size = 10 # 5 to 150
        self.otsu_bins = 800 # 2 to 1024

        # self.depth_axis = config.depth_axis # in line 49
        # self.clustering_mode = config.clustering
        # self.create_evaluation = config.create_evaluation
        # self.automatic_thresholding = config.automatic_thresholding
        # self.min_gap_volume = config.min_gap_volume
        # self.max_gap_volume = config.max_gap_volume
        # self.KM_number_of_clusters = config.KM_number_of_clusters
        # self.B_branching_factor = config.B_branching_factor
        # self.B_threshold = config.B_threshold
        # self.DB_eps = config.DB_eps
        # self.HDB_min_cluster_size = config.HDB_min_cluster_size
        # self.otsu_bins = config.otsu_bins

        self.gaps = []  #gap lsit

    def __str__(self):
        if self.gaps:
            return str(np.asanyarray(self.gaps)[:,4]) + "\nNext round!"

        else:
            return str("No Gaps Detected.")

    def threshold(self, points):
        # ----- AUTOMATIC THRESHOLDING TO FIND GAPS -----
       # points = helpers.VTX_to_numpy_array(points)
        depth_axis_pts = points[:,self.depth_axis]
        if self.automatic_thresholding == -1:
            self.surface_height = np.median(points[:, self.depth_axis])
            return points

        if(self.automatic_thresholding == 0):
            try:
                threshold = threshold_minimum(depth_axis_pts)
            except RuntimeError:
               raise Exception('Threshold_minimum was unable to find two maxima in histogram!')
        elif(self.automatic_thresholding == 1):
            threshold = threshold_li(depth_axis_pts)
        elif(self.automatic_thresholding == 2):
            threshold = threshold_yen(depth_axis_pts)
        elif(self.automatic_thresholding == 3):
            threshold = threshold_otsu(depth_axis_pts, self.otsu_bins)
        else:
            raise Exception('Automatic threshold value out of bounds!')

        print("Threshold is: ", threshold)

        device_surface_pts = points[depth_axis_pts <= threshold]
        self.surface_height = np.median(device_surface_pts[:, self.depth_axis])
        print("Device surfaceheight = ", self.surface_height)
        points = points[depth_axis_pts <= threshold]
        

        return points

    def clustering(self, points):
        print("clustering!")
        # ----- CLUSTERING THE GAPS -----
        clustering_switch = {
            0: self.kmeans,
            1: self.birch,
            2: self.dbscan,
            3: self.hdbscan
        }
        cluster_algorithm = clustering_switch[self.clustering_mode]
        labels = cluster_algorithm(points)

        labels = np.array(labels)
        labels_T = np.array([labels]).T
        clustered_points = np.append(points, labels_T, axis=1)


        clusters = []
        for i in set(labels):
            cluster = clustered_points[clustered_points[:, 3] == float(i)]
            cluster = cluster[:, [0, 1, 2]]

            # To construct a convex hull a minimum of 4 points is needed
            num_of_points, dim = cluster.shape
            if(num_of_points >= 4):
                clusters.append(cluster)

        
        print("num clusters: ", len(clusters), num_of_points)

        return (clusters, num_of_points)


# =============== MAIN DETECTION LOOP ===============
    def detector_callback(self, points):

        depth_axis_pts = points[:, self.depth_axis]
        print("depth_axis_pts", depth_axis_pts.shape)
        points = points[depth_axis_pts != 0]
        print("new points", points.shape)
        points = self.threshold(points) #! DISABLED
        print("new2 points", points.shape)
        clusters, num_of_points = self.clustering(points)
        # ----- VOLUME CORRECTION -----
        volume_corrected_clusters = []
        num_vol_corrected_pts = 0
        volume_corrected_pts_tuple = ()
        for cluster in clusters:
            hull = ConvexHull(cluster, qhull_options="QJ")

            # Map from vertex to point in cluster
            vertices = []
            for vertex in hull.vertices:
                x, y, z = cluster[vertex]
                vertices.append([x, y, z])

            gap = cluster.tolist()
            for vertex in vertices:
                # For each vertex, add a new point to the gap with the height
                # of the surface and the other axes corresponding to the vertex
                if(self.depth_axis == 0):
                    volume_point = [self.surface_height, vertex[1], vertex[2]]
                elif(self.depth_axis == 1):
                    volume_point = [vertex[0], self.surface_height, vertex[2]]
                elif(self.depth_axis == 2):
                    volume_point = [vertex[0], vertex[1], self.surface_height]

                gap.append(volume_point)
                num_vol_corrected_pts = num_vol_corrected_pts + 1

            volume_corrected_clusters.append(np.array(gap))
            volume_corrected_pts_tuple = volume_corrected_pts_tuple + \
                (num_vol_corrected_pts,)
            num_vol_corrected_pts = 0

        # ---- CALCULATING CONVEX HULLS OF GAPS AND THEIR CENTER -----
        self.convex_hulls_and_info = helpers.calculate_convex_hulls_and_centers(volume_corrected_clusters) 

        # ---- FILTER BASED ON VOLUME -----
        self.gaps = [] 
        for gap_info in self.convex_hulls_and_info:
            gap_volume = gap_info[4]
            if(self.min_gap_volume <= gap_volume <= self.max_gap_volume):
                self.gaps.append(gap_info)

        
        # ----- EVALUATION -----
        if(self.create_evaluation):
            num_of_points = np.subtract(num_of_points, num_vol_corrected_pts)
            helpers.evaluate_detector(num_of_points)
            self.create_evaluation = False

        self.detected_gap.set()  # signal succesful gap detection


    # =============== GAP DETECTION SERVICE ===============
    def gap_detection_call(self,depth_image):
        current_time = time.localtime()
        self.detected_gap.clear()
        self.detector_callback(depth_image) #pc is the depth frames pointcloud
        self.detected_gap.wait(3.)  # wait for 3 seconds

        if self.gaps is None:
            print('No gaps yet, skipping.')
            return []


    # =============== CLUSTER ALGORITHM WRAPPERS ===============
    def kmeans(self, data):
        return cluster.KMeans(n_clusters=self.KM_number_of_clusters).fit_predict(data)

    def birch(self, data):
        params = {'branching_factor': self.B_branching_factor,
                  'threshold': self.B_threshold,
                  'n_clusters': None,
                  'compute_labels': True}

        return cluster.Birch(**params).fit_predict(data)

    def dbscan(self, data):
        return cluster.DBSCAN(eps=self.DB_eps).fit_predict(data)

    def hdbscan(self, data):
        return hdbscan.HDBSCAN(min_cluster_size=self.HDB_min_cluster_size).fit_predict(data)


    