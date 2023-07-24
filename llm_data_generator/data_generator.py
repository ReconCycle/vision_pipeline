import os
import sys

from labelme_importer import LabelMeImporter


class LLMDataGenerator():
    """
    implement this
    generate data to train LLM
    """

    def __init__(self) -> None:
        pass

        # load list of images, with ground truths, labelme style
        # convert ground truths to Detections


        dataset_dir = os.path.expanduser("~/datasets2/reconcycle/2022-05-02_kalo_qundis/labelme")
        labelme_importer = LabelMeImporter()
        
        img_paths, all_detections, all_graph_relations = labelme_importer.process_labelme_dir(dataset_dir)

        # TODO: create graph relations
        # TODO: run action_prediction_decision_tree.py





if __name__ == '__main__':
    llm_data_generator = LLMDataGenerator()