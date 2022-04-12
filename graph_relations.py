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

from graph_tool.all import *

import matplotlib.pylab as plt
import networkx as nx


from enum import IntEnum

class Action(IntEnum):
    move = 0
    cut = 1
    lever = 2
    turn_over = 3
    remove_clip = 4

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
    if dist == 0 and (compute_is_inside(poly1, poly2) or compute_is_inside(poly2, poly1)):
        return False

    if dist < 10: # pixels
        return True


class GraphRelations:
    def __init__(self, class_names, classes, scores, boxes, masks, obb_boxes, obb_centers, tracking_ids, tracking_boxes, tracking_scores):
        self.class_names = class_names
        self.classes = classes
        self.obb_boxes = obb_boxes
        

        
    def using_network_x(self, save_file_path=None):
        class_names = self.class_names
        classes = self.classes
        obb_boxes = self.obb_boxes
        
        # class_objs = []
        # for a_class in classes:
            # a_class 
        # ! todo create objects for each node in graph
        
        ids = np.arange(len(classes))
        my_class_names = [class_names[i] for i in classes]
        print("my_class_names", my_class_names)
        
        inside_edges = []
        next_to_edges = []
        
        for id1, id2 in permutations(ids, 2): # we use permutations instead because not always associative
            poly1 = Polygon(obb_boxes[id1])
            poly2 = Polygon(obb_boxes[id2])
            inside = is_inside(poly1, poly2)
            next_to = compute_is_next_to(poly1, poly2)

            if inside:
                print(class_names[classes[id1]], id1, "inside", class_names[classes[id2]], id2)
                inside_edges.append((id1, id2))
            
            if next_to:
                print(class_names[classes[id1]], id1, "next to", class_names[classes[id2]], id2)
                next_to_edges.append((id1, id2))


        G = nx.DiGraph()
        
        plt.rcParams["figure.autolayout"] = True
        fig = plt.figure(1, figsize=(12, 9)) # , tight_layout={"pad": 20 }
        fig.clear(True)
        
        G.add_nodes_from(ids)
        G.add_edges_from(inside_edges + next_to_edges)

        pos = nx.spring_layout(G, k=2)
        
        print("node labels", {node: class_names[classes[node]] for node in G.nodes()})
        
        node_colors = []
        for id in ids:
            if class_names[classes[id]] == "battery":
                node_colors.append("blue")
            else:
                node_colors.append("pink")
        
        
        # add a margin
        ax1 = plt.subplot(111)
        ax1.margins(0.12)
        
        nx.draw(
            G, pos, ax=ax1, edge_color='black', width=1, linewidths=1,
            node_size=500, node_color=node_colors,
            labels={node: class_names[classes[node]] for node in G.nodes()},
            font_size=32
        )

        # for u, v, d in G.edges(data=True):
        #     d['weight'] = 30

        # edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
        # nx.draw(G, pos, node_color='b', edge_color=weights, width=2, with_labels=True)
        # nx.draw(G, pos)
        
        edge_labels_dict = {}
        for inside_edge in inside_edges:
            edge_labels_dict[inside_edge] = "in" # inside
            
        for next_to_edge in next_to_edges:
            edge_labels_dict[next_to_edge] = "next to" #nextto

        print("edge_labels_dict", edge_labels_dict)

        nx.draw_networkx_edge_labels(G, pos, edge_labels_dict,
                                    label_pos=0.5,
                                    font_size=32
                                    )
        
        io_buf = io.BytesIO()
        fig.savefig(io_buf, format='raw')
        io_buf.seek(0)
        img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                             newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1)) 
                            #newshape=(int(480), int(640), -1))
        io_buf.flush()
        io_buf.close()
        
        
        # ! todo, AI logic
        # if battery not connected to anything: pick up battery
        
        # def get_ids_with_class_name(class_name):
        #     ids_with_class_name = []
        #     for id in ids:
        #         if class_names[classes[id]] == "battery":
        #             ids_with_class_name.append(id)
        #     return ids_with_class_name
        
        for id in ids:
            if class_names[classes[id]] == "battery":
                descendents = nx.descendants(G, id)
                neighbours = nx.neighbors(G, id) # direct descendents
                if len(neighbours) == 0:
                    print("action: battery put in bin.")
                else:
                    for neighbour_id in neighbours:
                        class_name = class_names[classes[neighbour_id]]
                        
                        # todo: differentiate between in and next to.
                        if class_name == "hca_front":
                            print("action: turn over ", descendents)
                        if class_name == "pcb":
                            print("action: battery cut away from pcb")
                            
                        
                    print("action: separate battery from ...")
                    pass
                
        
        # if battery in graph:
        #     if battery is not connected to anything:
        #         print("action: put battery in bin.")
        
        # todo: define action(obj1, obj2, action_type)
        label = self.label
        
        def do_action(obj1, obj2, action_type):
            pass
        
        def is_inside(obj1, obj2):
            pass
            
        def is_next_to(obj1, obj2):
            pass
        
        def exists(obj):
            pass
        
        if is_inside(label.plastic_clip, label.hca_back):
            do_action(label.plastic_clip, label.hca_back, Action.remove_clip)
            
        elif is_next_to(label.battery, label.pcb):
            do_action(label.battery, label.pcb, Action.cut)
        
        elif is_inside(label.pcb, label.hca_back):
            do_action(label.pcb, label.hca_back, Action.lever)
            
        elif is_inside(label.pcb_covered, label.hca_back):
            do_action(label.pcb_covered, label.hca_back, Action.lever)
            
        elif exists(label.hca_front):
            do_action(label.hca_front, None, Action.turn_over)
        

        
        
        return img_arr[:, :, :3]
    