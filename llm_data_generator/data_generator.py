import os
import sys
from rich import print
import natsort

# ROS
from context_action_framework.types import Module

from labelme_importer import LabelMeImporter
from action_prediction_decision_tree import ActionPredictorDecisionTree


class LLMDataGenerator():
    """
    implement this
    generate data to train LLM
    """

    def __init__(self) -> None:


        self.labelme_importer = LabelMeImporter()
        self.action_prediction_decision_tree = ActionPredictorDecisionTree()
        # load list of images, with ground truths, labelme style
        # convert ground truths to Detections


        datasets_dir = os.path.expanduser("~/datasets2/reconcycle/2023-07-25_disassembly_sequences/")
        
        subfolders = [ f.path for f in os.scandir(datasets_dir) if f.is_dir() ]
        subfolders = natsort.os_sorted(subfolders)
        
        print("subfolders", subfolders)
        
        for subfolder in subfolders:
            
            print(f"\n[green]subfolder: {subfolder}")
        
            img_paths, all_detections, all_graph_relations, modules, cameras = self.labelme_importer.process_labelme_dir(subfolder)

            for img_path, detections, graph_relations, module, camera in zip(img_paths, all_detections, all_graph_relations, modules, cameras):
                # print("detections", detections)
                if module is None:
                    print("[red]Module is None!")
                action_type, action_block, reason = self.action_prediction_decision_tree.decision_tree(img_path, detections, graph_relations, module)
                
                graph_relations_text = self.graph_relations_to_text(graph_relations)
                action_text = self.action_to_text(action_type, action_block)
                
                print(graph_relations_text)
                
                if action_type is not action_type.none:
                    print(f"[green]{reason} -> {module.name}, {action_type.name}\n")
                else:
                    print(f"[red]{reason} -> {module.name}, {action_type.name}\n")
                    
                # TODO: generate LLM text.
                
                
                
            print("[red]DEBUG stopped iterating subfolders.")
            break #! DEBUG

    def graph_relations_to_text(self, graph_relations):
        """
        converts graph_relations to natural language
        """
        
        text_groups = ""
        text_single_items = ""
        single_items_list = []
        
        for idx, relations_group in enumerate(graph_relations.relations_groups):
            
            if relations_group: # check if there are relations
                text_groups += f"Group:\n"
                for relation_ids, relation_type in relations_group.items():
                    relation_id_1, relation_id_2 = relation_ids
                    relation1 = graph_relations.get_detection_by_id(relation_id_1)
                    relation2 = graph_relations.get_detection_by_id(relation_id_2)
                    text_groups += f"{relation1.label.name} [blue]{relation_type}[/blue] {relation2.label.name}\n"
        
            else:
                # group with 1 element, so no relations
                group = graph_relations.groups[idx]
                if len(group) == 1:
                    detection = group[0]
                    single_items_list.append(detection.label.name)
        
        # list also all the items that don't have relations
        if len(single_items_list) == 1:
            text_single_items = "Single component: " + ', '.join(single_items_list)
        elif len(single_items_list) > 1:
            text_single_items = "Single components: " + ', '.join(single_items_list)
        
        return text_groups + text_single_items
    
    def action_to_text(self, action_type, action_block):
        # TODO: convert action_type and action_block to text
        
        return "TODO"
    
    


if __name__ == '__main__':
    llm_data_generator = LLMDataGenerator()