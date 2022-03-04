import os
import numpy as np
import time
import cv2
from rich import print

from itertools import permutations
from shapely.geometry import Polygon

from graph_tool.all import *

def is_inside(poly1, poly2):
    """ is box1 inside box2? """
    
    intersect = poly1.intersection(poly2).area
    
    ratio_p1_inside_p2 = intersect / poly1.area

    
    print("intersect", intersect)
    print("ratio_p1_inside_p2", ratio_p1_inside_p2)
    
    # union = p1.union(p2).area
    # iou = intersect / union
    # print("union", union)
    # print("iou", iou)

    # if more than 90% of p1 is inside p2, then it is inside
    return ratio_p1_inside_p2 > 0.9


def is_next_to(poly1, poly2):
    dist = poly1.distance(poly2)
    if dist == 0 and (is_inside(poly1, poly2) or is_inside(poly2, poly1)):
        return False

    if dist < 10: # pixels
        return True


class GraphRelations:
    def __init__(self, class_names, classes, scores, boxes, masks, obb_boxes, obb_centers, tracking_ids, tracking_boxes, tracking_scores):

        ids = np.arange(len(classes))
        my_class_names = [class_names[i] for i in classes]
        print("my_class_names", my_class_names)
        
        g = Graph(directed=False)
        g.add_vertex(len(ids))
        e_inside = g.new_ep("int")
        e_next_to = g.new_ep("int")
        v_names = g.new_vertex_property("string")
        
        # ! working but commented out for now
        
        for i in ids:
            v_names[i] = my_class_names[i]
            print("class name", class_names[classes[i]])
            
        
        for id1, id2 in permutations(ids, 2): # we should use permutations instead because not alway associative
            poly1 = Polygon(obb_boxes[id1])
            poly2 = Polygon(obb_boxes[id2])
            inside = is_inside(poly1, poly2)
            next_to = is_next_to(poly1, poly2)
            print("is", class_names[classes[id1]], "inside", class_names[classes[id2]], "?", inside)
            print("is", class_names[classes[id1]], "next to", class_names[classes[id2]], "?", next_to)

            if inside or next_to:
                print("[id1, id2, inside, next_to]", [id1, id2, inside, next_to])
                g.add_edge_list([(id1, id2, int(inside), int(next_to))], eprops=[e_inside, e_next_to])
        
        # print("vertex classes", class_names[classes[g.vertex_index]])
        # print("vertex classes??", class_names[g.vertex_index])
        
        graph_draw(g, vertex_text=v_names, edge_text=e_next_to, vertex_font_size=18, edge_font_size=12)