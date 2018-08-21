import os
import numpy as np
from nilearn.image import new_img_like

from .utils.utils import resize, read_image_files
from .utils import crop_img, crop_img_to, read_image


def save_data_npy(npy_path, file_prefix, data=None, truth=None, affine=None):

    output_dir = os.path.dirname(os.path.join(npy_path, file_prefix + '_data.npy'))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if data is not None:
        np.save(os.path.join(npy_path, file_prefix + '_data.npy'),
                data.astype(np.float32))
    if truth is not None:
        np.save(os.path.join(npy_path, file_prefix + '_truth.npy'),
                truth.astype(np.uint8))
    if affine is not None:
        np.save(os.path.join(npy_path, file_prefix + '_affine.npy'),
                affine.astype(np.float32))


def load_data_npy(npy_path, file_prefix, data=False, truth=False, affine=False):
    if data:
        data = np.load(os.path.join(npy_path, file_prefix + '_data.npy'))
    if truth:
        truth = np.load(os.path.join(npy_path, file_prefix + '_truth.npy'))
    if affine:
        affine = np.load(os.path.join(npy_path, file_prefix + '_affine.npy'))
    return data, truth, affine


def write_data_to_file(npy_path, subject_ids, training_data_files, image_shape,
                       normalize=True, crop=True, overwrite=False):
    for index, set_of_files in enumerate(training_data_files):
        subject_id = subject_ids[index]
        subject_output_folder = os.path.dirname(os.path.join(npy_path, subject_id))
        if not os.path.exists(subject_output_folder) or overwrite:
            n_channels = len(set_of_files) - 1
            images = reslice_image_set(set_of_files, image_shape, label_indices=n_channels, crop=crop)
            subject_data = [image.get_data() for image in images]
            if normalize:
                subject_data = HU2uint8(subject_data)
            save_data_npy(npy_path, subject_id,
                          np.asarray(subject_data[:n_channels]),
                          np.asarray(subject_data[n_channels], dtype=np.uint8),
                          np.asarray(images[0].affine))

    # if normalize:
    #     normalize_all_data(npy_path, subject_ids)


def HU2uint8(image, HU_min=-1200.0, HU_max=600.0, HU_nan=-2000.0):
    image_new = np.array(image)
    image_new[np.isnan(image_new)] = HU_nan

    # normalize to [0, 1]
    image_new = (image_new - HU_min) / (HU_max - HU_min)
    image_new = np.clip(image_new, 0, 1)
    image_new = (image_new * 255).astype(np.float32)
    image_new = (image_new - 128.0) / 128.0
    return image_new

#
# def normalize_data(data, mean, std):
#     data -= mean[:, np.newaxis, np.newaxis, np.newaxis]
#     data /= std[:, np.newaxis, np.newaxis, np.newaxis]
#     return data
#
#
# def normalize_all_data(npy_path, subject_ids):
#     means = list()
#     stds = list()
#     for subject_id in subject_ids:
#         data, _, _ = load_data_npy(npy_path, subject_id, data=True, truth=False, affine=False)
#         means.append(data.mean(axis=(1, 2, 3)))
#         stds.append(data.std(axis=(1, 2, 3)))
#     mean = np.asarray(means).mean(axis=0)
#     std = np.asarray(stds).mean(axis=0)
#     for subject_id in subject_ids:
#         data, _, _ = load_data_npy(npy_path, subject_id, data=True, truth=False, affine=False)
#         save_data_npy(npy_path, subject_id, normalize_data(data, mean, std))


def find_downsized_info(training_data_files, input_shape):
    foreground = get_complete_foreground(training_data_files)
    crop_slices = crop_img(foreground, return_slices=True, copy=True)
    cropped = crop_img_to(foreground, crop_slices, copy=True)
    final_image = resize(cropped, new_shape=input_shape, interpolation="nearest")
    return crop_slices, final_image.affine, final_image.header


def get_cropping_parameters(in_files):
    if len(in_files) > 1:
        foreground = get_complete_foreground(in_files)
    else:
        foreground = get_foreground_from_set_of_files(in_files[0], return_image=True)
    return crop_img(foreground, return_slices=True, copy=True)


def reslice_image_set(in_files, image_shape, out_files=None, label_indices=None, crop=False):
    if crop:
        crop_slices = get_cropping_parameters([in_files])
    else:
        crop_slices = None
    images = read_image_files(in_files, image_shape=image_shape, crop=crop_slices, label_indices=label_indices)
    if out_files:
        for image, out_file in zip(images, out_files):
            image.to_filename(out_file)
        return [os.path.abspath(out_file) for out_file in out_files]
    else:
        return images


def get_complete_foreground(training_data_files):
    for i, set_of_files in enumerate(training_data_files):
        subject_foreground = get_foreground_from_set_of_files(set_of_files)
        if i == 0:
            foreground = subject_foreground
        else:
            foreground[subject_foreground > 0] = 1

    return new_img_like(read_image(training_data_files[0][-1]), foreground)


def get_foreground_from_set_of_files(set_of_files, background_value=0, tolerance=0.00001, return_image=False):
    for i, image_file in enumerate(set_of_files):
        image = read_image(image_file)
        is_foreground = np.logical_or(image.get_data() < (background_value - tolerance),
                                      image.get_data() > (background_value + tolerance))
        if i == 0:
            foreground = np.zeros(is_foreground.shape, dtype=np.uint8)

        foreground[is_foreground] = 1
    if return_image:
        return new_img_like(image, foreground)
    else:
        return foreground
