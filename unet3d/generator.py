import os
import copy
import itertools
import numpy as np
from random import shuffle


from .utils import pickle_dump, pickle_load
from .utils.patches import compute_patch_indices, get_random_nd_index, get_patch_from_3d_data
from .augment import augment_data, random_permutation_x_y
from .data import load_data_npy


def get_training_and_validation_generators(npy_path, subject_ids_file,
                                           batch_size, validation_batch_size,
                                           n_labels, labels,
                                           training_keys_file, validation_keys_file, data_split=0.8, overwrite=False,
                                           augment=False, augment_flip=True,
                                           augment_distortion_factor=0.25, permute=False,
                                           image_shape=None, patch_shape=None,
                                           validation_patch_overlap=0, training_patch_start_offset=None,
                                           skip_blank=True):
    subject_ids = pickle_load(subject_ids_file)

    training_list, validation_list = get_training_and_validation_split(len(subject_ids),
                                                                       training_file=training_keys_file,
                                                                       validation_file=validation_keys_file,
                                                                       data_split=data_split,
                                                                       overwrite=overwrite)
    print('training_list', len(training_list))
    print('validation_list', len(validation_list))
    print("get training generator")
    training_generator = data_generator(npy_path, subject_ids, training_list,
                                        batch_size=batch_size,
                                        n_labels=n_labels,
                                        labels=labels,
                                        augment=augment,
                                        augment_flip=augment_flip,
                                        augment_distortion_factor=augment_distortion_factor,
                                        permute=permute,
                                        image_shape=image_shape,
                                        patch_shape=patch_shape,
                                        patch_overlap=0,
                                        patch_start_offset=training_patch_start_offset,
                                        skip_blank=skip_blank)
    print("get validation generator")
    validation_generator = data_generator(npy_path, subject_ids, validation_list,
                                          batch_size=validation_batch_size,
                                          n_labels=n_labels,
                                          labels=labels,
                                          image_shape=image_shape,
                                          patch_shape=patch_shape,
                                          patch_overlap=validation_patch_overlap,
                                          skip_blank=skip_blank,
                                          shuffle_list=False)

    # Set the number of training and validation samples per epoch correctly
    num_training_steps = get_number_of_steps(get_number_of_patches(npy_path, subject_ids, training_list,
                                                                   image_shape=image_shape,
                                                                   patch_shape=patch_shape,
                                                                   patch_overlap=0,
                                                                   patch_start_offset=training_patch_start_offset,
                                                                   skip_blank=skip_blank),
                                             batch_size)
    print("Number of training steps: ", num_training_steps)

    num_validation_steps = get_number_of_steps(get_number_of_patches(npy_path, subject_ids, validation_list,
                                                                     image_shape=image_shape,
                                                                     patch_shape=patch_shape,
                                                                     patch_overlap=validation_patch_overlap,
                                                                     skip_blank=skip_blank),
                                               validation_batch_size)
    print("Number of validation steps: ", num_validation_steps)

    return training_generator, validation_generator, num_training_steps, num_validation_steps


def get_training_and_validation_split(n_samples, training_file, validation_file, data_split=0.8, overwrite=False):
    if overwrite or not os.path.exists(training_file):
        print("Creating training and validation split...")
        sample_list = list(range(n_samples))
        training_list, validation_list = split_list(sample_list, split=data_split)
        pickle_dump(training_list, training_file)
        pickle_dump(validation_list, validation_file)
        return training_list, validation_list
    else:
        print("Loading previous training and validation split...")
        return pickle_load(training_file), pickle_load(validation_file)


def split_list(input_list, split=0.8, shuffle_list=True):
    if shuffle_list:
        shuffle(input_list)
    n_training = int(len(input_list) * split)
    training = input_list[:n_training]
    validation = input_list[n_training:]
    return training, validation


def data_generator(npy_path, subject_ids, index_list, batch_size=1, n_labels=1, labels=None,
                   augment=False, augment_flip=True, augment_distortion_factor=0.25, permute=False,
                   image_shape=None, patch_shape=None,
                   patch_overlap=0, patch_start_offset=None,
                   skip_blank=True, shuffle_list=True):
    orig_index_list = index_list
    while True:
        x_list = list()
        y_list = list()
        if patch_shape:
            index_list = create_patch_index_list(orig_index_list,
                                                 image_shape, patch_shape,
                                                 patch_overlap, patch_start_offset)
        else:
            index_list = copy.copy(orig_index_list)

        if shuffle_list:
            shuffle(index_list)
        while len(index_list) > 0:
            index = index_list.pop()
            add_data(x_list, y_list, npy_path, subject_ids, index,
                     augment=augment, augment_flip=augment_flip,
                     augment_distortion_factor=augment_distortion_factor, permute=permute,
                     patch_shape=patch_shape,
                     skip_blank=skip_blank)
            if len(x_list) == batch_size or (len(index_list) == 0 and len(x_list) > 0):
                yield convert_data(x_list, y_list, n_labels=n_labels, labels=labels)
                x_list = list()
                y_list = list()


def create_patch_index_list(index_list, image_shape, patch_shape, patch_overlap, patch_start_offset=None):
    patch_index = list()
    for index in index_list:
        if patch_start_offset is not None:
            random_start_offset = np.negative(get_random_nd_index(patch_start_offset))
            patches = compute_patch_indices(image_shape, patch_shape, overlap=patch_overlap, start=random_start_offset)
        else:
            patches = compute_patch_indices(image_shape, patch_shape, overlap=patch_overlap)
        patch_index.extend(itertools.product([index], patches))
    return patch_index


def add_data(x_list, y_list, npy_path, subject_ids, index,
             augment=False, augment_flip=False, augment_distortion_factor=0.25, permute=False,
             patch_shape=False, skip_blank=True):
    data, truth = get_data_from_file(npy_path, subject_ids, index, patch_shape=patch_shape)
    if augment:
        if patch_shape is not None:
            _, _, affine = load_data_npy(npy_path, subject_ids[index[0]], affine=True)
        else:
            _, _, affine = load_data_npy(npy_path, subject_ids[index], affine=True)
        data, truth = augment_data(data, truth, affine, flip=augment_flip, scale_deviation=augment_distortion_factor)

    truth = truth[np.newaxis]
    if permute:
        if data.shape[-3] != data.shape[-2] or data.shape[-2] != data.shape[-1]:
            raise ValueError("To utilize permutations, data array must be in 3D cube shape with all dimensions having "
                             "the same length.")
        data, truth = random_permutation_x_y(data, truth)

    if not skip_blank or np.any(truth != 0):
        x_list.append(HU2uint8(data))
        y_list.append(truth)


def HU2uint8(image, HU_min=-1200.0, HU_max=600.0, HU_nan=-2000.0):
    image_new = np.array(image)
    image_new[np.isnan(image_new)] = HU_nan

    # normalize to [0, 1]
    image_new = (image_new - HU_min) / (HU_max - HU_min)
    image_new = np.clip(image_new, 0, 1)
    image_new = (image_new * 255).astype(np.float32)
    image_new = (image_new - 128.0) / 128.0
    return image_new


def get_data_from_file(npy_path, subject_ids, index, patch_shape=None):
    if patch_shape:
        index, patch_index = index
        data, truth = get_data_from_file(npy_path, subject_ids, index, patch_shape=None)
        x = get_patch_from_3d_data(data, patch_shape, patch_index)
        y = get_patch_from_3d_data(truth, patch_shape, patch_index)
    else:
        x, y, _ = load_data_npy(npy_path, subject_ids[index], data=True, truth=True)
    return x, y


def convert_data(x_list, y_list, n_labels=1, labels=None):
    x = np.asarray(x_list)
    y = np.asarray(y_list)
    if n_labels == 1:
        y[y > 0] = 1
    elif n_labels > 1:
        y = get_multi_class_labels(y, n_labels=n_labels, labels=labels)
    return x, y


def get_multi_class_labels(data, n_labels, labels=None):
    new_shape = [data.shape[0], n_labels] + list(data.shape[2:])
    y = np.zeros(new_shape, np.int8)
    for label_index in range(n_labels):
        if labels is not None:
            y[:, label_index][data[:, 0] == labels[label_index]] = 1
        else:
            y[:, label_index][data[:, 0] == (label_index + 1)] = 1
    return y


def get_number_of_steps(n_samples, batch_size):
    if n_samples <= batch_size:
        return n_samples
    elif np.remainder(n_samples, batch_size) == 0:
        return n_samples // batch_size
    else:
        return n_samples // batch_size + 1


def get_number_of_patches(npy_path, subject_ids, index_list, image_shape=None, patch_shape=None,
                          patch_overlap=0, patch_start_offset=None,
                          skip_blank=True):
    # if patch_shape:
    #     index_list = create_patch_index_list(index_list,
    #                                          image_shape, patch_shape,
    #                                          patch_overlap, patch_start_offset)
    #     count = 0
    #     for index in index_list:
    #         x_list = list()
    #         y_list = list()
    #         add_data(x_list, y_list, npy_path, subject_ids, index, patch_shape=patch_shape, skip_blank=skip_blank)
    #         count += len(x_list)
    #     return count
    # else:
    #     return len(index_list)
    return len(index_list)
