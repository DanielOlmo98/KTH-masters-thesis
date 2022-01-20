import numpy as np
import itertools
import utils
import os
from medpy.io.load import load


def load_patient_data(patient_path):
    path = utils.get_project_root() + patient_path
    file_list = os.listdir(path)
    images = []
    segmentations = []
    headers = []
    for file_pair in file_list:
        common_file_path = path + file_pair
        if not os.path.isfile(common_file_path):
            continue
        if "sequence" in file_pair:
            continue
        if file_pair[-4:] == ".mhd":
            if file_pair[-6:] == "gt.mhd":
                img, header = load(common_file_path)
                segmentations.append(img)
            else:
                img, header = load(common_file_path)
                images.append(img)
                headers.append(header)
    return zip(images, segmentations, headers)


if __name__ == '__main__':
    patient1 = load_patient_data('/dataset/training/patient0001/')
    for img, segmentation, header in patient1:
        if np.shape(img)[-1] == 1:
            utils.plot_image_g(img, overlay_img=segmentation, alpha_overlay=0.2)
