import os
import json
import albumentations as albu


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
            
def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')
            
def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        # albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor),
    ]
    return albu.Compose(_transform)