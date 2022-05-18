import os
from telnetlib import GA
from matplotlib.pyplot import tight_layout
import numpy as np
import time
import cv2
from rich import print
import io

from itertools import permutations
from shapely.geometry import Polygon

# from graph_tool.all import *

import matplotlib.pylab as plt
import networkx as nx

from typing import List
from helpers import Detection, Action


def compute_is_inside(poly1, poly2):
    """ is box1 inside box2? """
    
    intersect = poly1.intersection(poly2).area
    
    # stop division by 0
    if poly1.area < 0.001:
        return False
    
    ratio_p1_inside_p2 = intersect / poly1.area

    
    # print("intersect", intersect)
    # print("ratio_p1_inside_p2", ratio_p1_inside_p2)
    
    # union = p1.union(p2).area
    # iou = intersect / union
    # print("union", union)
    # print("iou", iou)

    # if more than 90% of p1 is inside p2, then it is inside
    return ratio_p1_inside_p2 > 0.9


def compute_is_next_to(poly1, poly2):
    dist = poly1.distance(poly2)
    # print("dist", dist)
    if dist == 0 and (compute_is_inside(poly1, poly2) or compute_is_inside(poly2, poly1)):
        return False

    if dist < 10: # pixels
        return True


class GraphRelations:
    def __init__(self, labels, detections: List[Detection]): 
        self.labels = labels
        self.detections = detections
        
    def using_network_x(self, save_file_path=None):
        labels = self.labels
        
        for detection in self.detections:
            
            poly = None
            if len(detection.mask_contour) > 2:
                poly = Polygon(detection.mask_contour)
                
            if poly is None or not poly.is_valid:
                poly = Polygon(detection.obb_corners)

            detection.mask_polygon = poly
        
        inside_edges = []
        next_to_edges = []        
        for detection1, detection2 in permutations(self.detections, 2):
            
            inside = compute_is_inside(detection1.mask_polygon, detection2.mask_polygon) # not associative
            next_to = compute_is_next_to(detection1.mask_polygon, detection2.mask_polygon) # associative

            if inside:
                # print(detection1.label, "inside", detection2.label)
                inside_edges.append((detection1.id, detection2.id))
            
            if next_to:
                # print(detection1.label, "next to", detection2.label)
                next_to_edges.append((detection1.id, detection2.id))

        edge_labels_dict = {}
        for inside_edge in inside_edges:
            edge_labels_dict[inside_edge] = "in" # inside
            
        for next_to_edge in next_to_edges:
            edge_labels_dict[next_to_edge] = "next to" #nextto

        node_colors = []
        for detection in self.detections:
            if detection.label == self.labels.battery:
                node_colors.append("blue")
            else:
                node_colors.append("pink")

        # BEGIN graph drawing
        G = nx.DiGraph()
        plt.rcParams["figure.autolayout"] = True
        fig = plt.figure(1, figsize=(12, 9)) # , tight_layout={"pad": 20 }
        fig.clear(True)
        
        G.add_nodes_from([detection.id for detection in self.detections])
        G.add_edges_from(inside_edges + next_to_edges)

        pos = nx.spring_layout(G, k=2)
        
        # print("edge_labels_dict", edge_labels_dict)
        # print("node labels", {id: self.detections[id].label.name for id in G.nodes()})
        
        # add a margin
        ax1 = plt.subplot(111)
        ax1.margins(0.12)
        
        nx.draw(
            G, pos, ax=ax1, edge_color='black', width=1, linewidths=1,
            node_size=500, node_color=node_colors,
            labels={id: self.detections[id].label.name for id in G.nodes()},
            font_size=32
        )

        nx.draw_networkx_edge_labels(G, pos, edge_labels_dict,
                                    label_pos=0.5,
                                    font_size=32
                                    )
        
        io_buf = io.BytesIO()
        fig.savefig(io_buf, format='raw')
        io_buf.seek(0)
        img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                             newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
        io_buf.flush()
        io_buf.close()
        # END graph drawing
        
        
        def do_action(det1=None, det2=None, action_type=None):
            sentence = ""
            def label(det):
                return det.label.name + " (" + str(det.tracking_id) + ")"
            
            if action_type == Action.move and det2 is None:
                sentence = "action: move " + label(det1)
            elif action_type == Action.turn_over and det2 is None:
                sentence = "action: turn over " + label(det1)
            elif action_type == Action.cut:
                sentence = "action: cut " + label(det1) + " away from " + label(det2)
            elif action_type == Action.lever:
                sentence = "action: lever " + label(det1) + " away from " + label(det2)
            elif action_type == Action.remove_clip:
                sentence = "action: remove clip " + label(det1) + " from " + label(det2)
            elif det1 == None and det2 == None and action_type == None:
                sentence = "action: undefined"
            else:
                sentence = "action: unknown"
                
            return det1, det2, action_type, sentence
        
        def is_inside(label1, label2):
            for key, val in edge_labels_dict.items():
                a, b = key
                detection_pair = (self.detections[a].label.value, self.detections[b].label.value)
                if detection_pair == (label1.value, label2.value) and val == "in":
                    return True, self.detections[a], self.detections[b]
                
            return False, None, None
            
        def is_next_to(label1, label2):
            for key, val in edge_labels_dict.items():
                a, b = key
                detection_pair = (self.detections[a].label.value, self.detections[b].label.value)
                if detection_pair == (label1.value, label2.value) and val == "next to":
                    return True, self.detections[a], self.detections[b]

            return False, None, None
        
        def exists(label):
            for detection in self.detections:
                if detection.label == label:
                    return True, detection
            
            return False, None
        
        # AI logic
        # todo: if a battery is present, remove all other connected subgraphs
        def ai():
            
            is_inside_bool, det1, det2 = is_inside(labels.plastic_clip, labels.hca_back)
            if is_inside_bool:
                return do_action(det1, det2, Action.remove_clip)
            
            is_inside_bool, det1, det2 = is_inside(labels.pcb, labels.hca_back)
            if is_inside_bool:
                return do_action(det1, det2, Action.lever)
                
            is_inside_bool, det1, det2 = is_inside(labels.pcb_covered, labels.hca_back)
            if is_inside_bool:
                return do_action(det1, det2, Action.lever)
            
            is_next_to_bool, det1, det2 = is_next_to(labels.battery, labels.pcb)
            if is_next_to_bool:
                return do_action(det1, det2, Action.cut)
            
            is_next_to_bool, det1, det2 = is_next_to(labels.battery, labels.pcb_covered)
            if is_next_to_bool:
                return do_action(det1, det2, Action.cut)
            
            exists_bool, det = exists(labels.hca_front)
            if exists_bool:
                return do_action(det, None, Action.turn_over)
            
            return do_action()
        
        action = ai()
        
        return img_arr[:, :, :3], action
    