import numpy as np
import utils
import os
import matplotlib as mpl
from medpy.io.load import load


def load_patient_data(patient_path):
    path = utils.get_project_root() + patient_path
    file_list = os.listdir(path)
    images = []
    segmentations = []
    headers = []
    sequences = []
    sequence_headers = []
    for file_pair in file_list:
        common_file_path = path + file_pair
        if not os.path.isfile(common_file_path):
            continue
        if file_pair[-4:] == ".mhd":
            if file_pair[-6:] == "gt.mhd":
                img, header = load(common_file_path)
                segmentations.append(img)
            elif "sequence" in file_pair:
                img, header = load(common_file_path)
                sequences.append(img)
                sequence_headers.append(header)
                continue
            else:
                img, header = load(common_file_path)
                images.append(img)
                headers.append(header)
    return zip(images, segmentations, headers), zip(sequences, sequence_headers)




if __name__ == '__main__':
    patient1, patient1_sequence = load_patient_data('/dataset/training/patient0001/')
    for img, segmentation, header in patient1:
        if np.shape(img)[-1] == 1:
            print(np.shape(img))
            utils.plot_image_g(np.squeeze(img), overlay_img=segmentation, alpha_overlay=0.2)
    # for volume, header in patient1_sequence:
    #     utils.slice_view_3d(volume)
