import os
import json


def load_data_and_labels(root:str, filename_lst:list, labels_lst:list )->None:
    """Load image and masks filenames from train folder to the list alongwith labels"""
    folder_names = os.listdir(root)
    # Opening JSON file
    with open('class_labels.json') as json_file:
        class_labels_dict = json.load(json_file)

    for folder in folder_names:
        filenames = os.listdir(os.path.join(root, folder))
        for filename in filenames:
            filename_lst.append(os.path.join(root, folder, filename))
            labels_lst.append(int(class_labels_dict[folder]))