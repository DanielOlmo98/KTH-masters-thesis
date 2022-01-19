import utils
import os
from medpy.io.load import load


def load_patient_data(patient_path):
    path = utils.get_project_root() + patient_path
    file_list = os.listdir(path)
    images = []
    headers = []
    for file in file_list:
        common_file_path = path + file
        if not os.path.isfile(common_file_path):
            continue
        if file[-4:] == ".mhd":
            img, header = load(common_file_path)
            images.append(img)
            headers.append(header)

    return zip(images, headers)


if __name__ == '__main__':
    patient1 = load_patient_data('/dataset/training/patient0001/')
    for img, header in patient1:
        utils.plot_image_g(img)
