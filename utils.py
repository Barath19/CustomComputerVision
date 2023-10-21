from typing import Union, List
import os
import numpy as np
import shutil
import splitfolders

def classification_segregator(images_path: Union[str, os.PathLike], 
                              output_path: Union[str, os.PathLike], 
                              labels_path: Union[str, os.PathLike],
                              yolo_format: bool) -> None:
    """
    Segregates images based on their labels.

    Parameters:
    - images_path: Path to the directory containing the images.
    - output_path: Path to the directory where the segregated images will be stored.
    - labels_path: Path to the file containing labels.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    else:
        shutil.rmtree(output_path)           # Removes all the subdirectories!
        os.makedirs(output_path)

    all_images_path = np.array([os.path.join(images_path, x) for x in sorted(list(os.listdir(images_path)))])

    # Read labels from the file
    with open(labels_path, 'r') as labels_file:
        # Assuming each line in the file corresponds to a label
        labels = np.array([line.strip() for line in labels_file])

    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        files = all_images_path[np.where(labels==label)]
        class_folder = os.path.join(output_path, label)
        os.makedirs(class_folder)

        for file in files:
            shutil.copy(file, os.path.join(class_folder, file.split(f'{os.path.sep}')[-1]) )

    if yolo_format:
        
        # Removes all the subdirectories!
        if not os.path.exists(os.path.abspath(output_path)+'_yolo'):
            os.makedirs(os.path.abspath(output_path)+'_yolo')
        else:
            shutil.rmtree(os.path.abspath(output_path)+'_yolo')           # Removes all the subdirectories!
            os.makedirs(os.path.abspath(output_path)+'_yolo')

        splitfolders.ratio(output_path, output=os.path.abspath(output_path)+'_yolo', seed=1337, ratio=(.8, .1, .1), group_prefix=None, move=False)


    

# Example usage:
# classification_segregator("path/to/images", "path/to/output", "path/to/labels.txt", "string_label")

classification_segregator('/home/bk/Research/competition/cloudfight/38th/level2/level_02/train_data','dataset/output','/home/bk/Research/competition/cloudfight/38th/level2/level_02/train_data_labels.csv', True)
