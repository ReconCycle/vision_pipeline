
import numpy as np
from scipy.spatial import ConvexHull
import os

def VTX_to_numpy_array(vtx):
    point_list = []
    for p in vtx:
        point_list.append([p[0], p[1], p[2]])

    return np.array(point_list)

def vtx_map_to_tex(vtx,tex):
    point_list = []
    for p in zip(vtx,tex):
        
        point_list.append([p[0][0],p[0][1], p[0][2]])

    return np.array(point_list)


def add_x_and_y(deptharray): # Shape of deptharray : [[z0.0, ... zx.0 ],...,[zy.0,...,zy.x]] (row by row from left to right)
    y_size,x_size = deptharray.shape
    # Numpy array of all points of the pointcloud given in order:
    #[[x1,y1,z1], [x2,y2,z2], ....
    points = []
    for y in range(y_size):
        for x in range(x_size):
            points.append([x, y, deptharray[y][x]])
            print(deptharray[y][x])
    return np.asanyarray(points)


def calculate_convex_hulls_and_centers(gaps) :
    """
    Calculates the convex hull for the given gap points including the
    simplices, vertices, volume, number of points in the gap and the center
    point(center of the hull).

    Parameters
    ----------
    gaps : list
        List of gaps with the points of the gap in order:
        [[[x1,y1,z1], [x2,y2,z2], ... ],[[x11,y11,z11], [x12,y12,z12], ...]...]
                      GAP1                              GAP2            ...

    Returns
    -------
    list
        List of convex hull info for each gap:
        [[center, vertices, simplices, hull.volume, sizes, num_of_pts],      GAP1
         [center, vertices, simplices, hull.volume, sizes, num_of_pts],      GAP2
                            ....                                      ....

    """

    convex_hull = []

    for gap in gaps:
        hull = ConvexHull(gap, qhull_options="QJ")

        # x, y, z components of the hull vertices
        vertices_x = hull.points[hull.vertices, 0]
        vertices_y = hull.points[hull.vertices, 1]
        vertices_z = hull.points[hull.vertices, 2]

        # Image Vertices
  
        img_vertices = np.asanyarray([v for v in zip(vertices_x.astype(int),vertices_y.astype(int),vertices_z.astype(int))])

        # calculate center of hull by taking middle of extremas of axes
        cx = (np.max(vertices_x)+np.min(vertices_x))/2
        cy = (np.max(vertices_y)+np.min(vertices_y))/2
        cz = (np.max(vertices_z)+np.min(vertices_z))/2
        center = [cx, cy, cz]

        # calculate the sizes
        sx = (np.max(vertices_x)-np.min(vertices_x))/2
        sy = (np.max(vertices_y)-np.min(vertices_y))/2
        sz = (np.max(vertices_x)-np.min(vertices_x))/2
        size = [sx, sy, sz]


        # map vertices and simplices to gap points
        vertices = []
        for vertex in hull.vertices:
            x, y, z = hull.points[vertex]
            vertices.append([x, y, z])

        simplices = []
        for simplex in hull.simplices:
            x, y, z = hull.points[simplex]
            simplices.append([x, y, z])

        # number of points(for evaluation)
        num_of_points, dim = gap.shape

        info = [center, vertices, img_vertices, simplices, hull.volume*1000000, size, num_of_points]
        convex_hull.append(info)

    return convex_hull



def evaluate_detector(num_of_points):
    """
    Evaluate the detector by writing infos about the found gaps to
    detector_evaluation.txt.

    Parameters
    ----------
    num_of_points : tuple
        Tuple of the number of points for each gap.

    """

    # rospack = rospkg.RosPack()
    # src_dir = rospack.get_path('ugoe_gap_detection_ros')
    # file_path = src_dir + "/evaluation/detector_evaluation.txt"

    src_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(src_dir,"/evaluation/detector_evaluation.txt" )

    try:
        file = open(file_path, "w")
    except (IOError, OSError):
        Exception("Could not open/read file:" + file_path)
        

    for gap, number_of_points in enumerate(num_of_points):
        file.write("Gap " + str(int(gap)+1) + " - " +
                   str(number_of_points) + " points\n")

    print("Saved evaluation report of detector to " + src_dir + "/evaluation/detector_evaluation.txt.")

    file.close()