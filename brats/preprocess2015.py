import os
import glob
import SimpleITK as sitk

from unet3d.utils import pickle_dump
from unet3d.data import write_data_to_file


pre_config = dict()
pre_config["original_modalities"] = ["_T1", "_T1c", "_Flair", "_T2", '.XX.*.OT.']
pre_config["all_modalities"] = ["t1", "t1Gd", "flair", "t2", "truth"]
pre_config["subject_ids_file"] = os.path.abspath("subject_ids.pkl")
pre_config["data_original"] = "data/original"
pre_config["data_preprocessed"] = "data/preprocessed"
pre_config["npy_path"] = "data/npy_data"
pre_config["image_shape"] = (128, 128, 128)  # This determines what shape the images will be cropped/resampled to.


def write_file(input_file, output_file):
    image = sitk.ReadImage(input_file)
    sitk.WriteImage(image, output_file)


def get_subject_modality(subject_folder, name):
    path = os.path.join(subject_folder, "*" + name + "*")
    try:
        subject_modality_path = glob.glob(path)[0]
        if os.path.isdir(subject_modality_path):
            return os.path.basename(subject_modality_path)
    except IndexError:
        raise RuntimeError("Could not find file matching {}".format(path))
    print('get_subject_modality error', subject_folder, name)


def convert_brats_folder(subject_id, input_folder, output_folder):
    for index, name in enumerate(pre_config["original_modalities"]):
        subject_modality = get_subject_modality(input_folder, name)
        input_file = os.path.abspath(os.path.join(input_folder,
                                                  subject_modality + '/' + subject_modality + '.mha'))
        output_file = os.path.abspath(os.path.join(output_folder,
                                                   subject_id + '_' + pre_config["all_modalities"][index] + ".nii.gz"))
        write_file(input_file, output_file)


def convert_brats_data(input_folder, output_folder, overwrite=False):
    for subject_folder in glob.glob(os.path.join(input_folder, "*", "*")):
        print('find subject', subject_folder)
        if not os.path.isdir(subject_folder):
            continue
        subject_id = os.path.basename(subject_folder)
        subject_category = os.path.basename(os.path.dirname(subject_folder))
        subject_output_folder = os.path.join(output_folder, subject_category, subject_id)
        if not os.path.exists(subject_output_folder) or overwrite:
            if not os.path.exists(subject_output_folder):
                os.makedirs(subject_output_folder)
            convert_brats_folder(subject_id, subject_folder, subject_output_folder)


def fetch_training_data_files(input_folder):
    training_data_files = list()
    subject_ids = list()
    for subject_folder in glob.glob(os.path.join(input_folder, "*", "*")):
        print('find subject', subject_folder)
        if not os.path.isdir(subject_folder):
            continue
        subject_id = os.path.basename(subject_folder)
        subject_category = os.path.basename(os.path.dirname(subject_folder))
        subject_ids.append(os.path.join(subject_category, subject_id))
        subject_files = list()
        for modality in pre_config["all_modalities"]:
            subject_files.append(os.path.join(subject_folder, subject_id + "_" + modality + ".nii.gz"))
        training_data_files.append(tuple(subject_files))
    return subject_ids, training_data_files


def main():
    # convert_brats_data(pre_config["data_original"], pre_config["data_preprocessed"], overwrite=False)

    subject_ids, training_files = fetch_training_data_files(pre_config["data_preprocessed"])
    pickle_dump(subject_ids, pre_config["subject_ids_file"])

    write_data_to_file(pre_config["npy_path"], subject_ids, training_files, image_shape=pre_config["image_shape"])


if __name__ == "__main__":
    main()
