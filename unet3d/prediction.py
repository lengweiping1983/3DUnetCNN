import os
import numpy as np

from .training import load_old_model
from .utils import pickle_load, get_image
from .utils.patches import reconstruct_from_patches, get_patch_from_3d_data, compute_patch_indices
from .augment import permute_data, generate_permutation_keys, reverse_permute_data
from .data import load_data_npy


def run_validation_cases(model_file, training_modalities,
                         validation_keys_file, npy_path, subject_ids_file,
                         label_map=False, output_dir=".",
                         threshold=0.5, labels=None, overlap=16, permute=False):
    model = load_old_model(model_file)
    validation_indices = pickle_load(validation_keys_file)
    subject_ids = pickle_load(subject_ids_file)

    for index in validation_indices:
        print('predict', subject_ids[index])
        case_directory = os.path.join(output_dir, subject_ids[index])
        data, truth, affine = load_data_npy(npy_path, subject_ids[index], data=True, truth=True, affine=True)
        run_validation_case(model, training_modalities,
                            data, truth, affine,
                            label_map, case_directory,
                            threshold=threshold, labels=labels, overlap=overlap, permute=permute)


def run_validation_case(model, training_modalities,
                        data, truth, affine,
                        label_map, output_dir,
                        threshold=0.5, labels=None, overlap=16, permute=False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    test_data = np.asarray([data])
    for i, modality in enumerate(training_modalities):
        image = get_image(test_data[0, i], affine)
        image.to_filename(os.path.join(output_dir, "data_{0}.nii.gz".format(modality)))

    test_truth = get_image(truth, affine)
    test_truth.to_filename(os.path.join(output_dir, "truth.nii.gz"))

    patch_shape = tuple([int(dim) for dim in model.input.shape[-3:]])
    if patch_shape == test_data.shape[-3:]:
        prediction = predict(model, test_data, permute=permute)
    else:
        prediction = patch_wise_prediction(model=model, data=test_data, overlap=overlap, permute=permute)
    prediction_image = prediction_to_image(prediction, affine, label_map=label_map,
                                           threshold=threshold, labels=labels)
    if isinstance(prediction_image, list):
        for i, image in enumerate(prediction_image):
            image.to_filename(os.path.join(output_dir, "prediction_{0}.nii.gz".format(i + 1)))
    else:
        prediction_image.to_filename(os.path.join(output_dir, "prediction.nii.gz"))


def predict(model, data, permute=False):
    if permute:
        predictions = list()
        for batch_index in range(data.shape[0]):
            predictions.append(predict_with_permutations(model, data[batch_index]))
        return np.asarray(predictions)
    else:
        return model.predict(data)


def predict_with_permutations(model, data):
    predictions = list()
    for permutation_key in generate_permutation_keys():
        temp_data = permute_data(data, permutation_key)
        predict_data = model.predict(temp_data[np.newaxis])[0]
        temp_data = reverse_permute_data(predict_data, permutation_key)
        predictions.append(temp_data)
    return np.mean(predictions, axis=0)


def prediction_to_image(prediction, affine, label_map=False, threshold=0.5, labels=None):
    if prediction.shape[1] == 1:
        data = prediction[0, 0]
        if label_map:
            label_map_data = np.zeros(data.shape, np.int8)
            if labels:
                label = labels[0]
            else:
                label = 1
            label_map_data[data > threshold] = label
            data = label_map_data
    elif prediction.shape[1] > 1:
        if label_map:
            label_map_data = get_prediction_labels(prediction, threshold=threshold, labels=labels)
            data = label_map_data[0]
        else:
            return multi_class_prediction(prediction, affine)
    else:
        raise RuntimeError("Invalid prediction array shape: {0}".format(prediction.shape))
    return get_image(data, affine)


def get_prediction_labels(prediction, threshold=0.5, labels=None):
    n_samples = prediction.shape[0]
    label_arrays = []
    for sample_number in range(n_samples):
        label_data = np.argmax(prediction[sample_number], axis=0) + 1
        label_data[np.max(prediction[sample_number], axis=0) < threshold] = 0
        if labels:
            for value in np.unique(label_data).tolist():
                if value == 0:
                    continue
                label_data[label_data == value] = labels[value - 1]
        label_arrays.append(np.array(label_data, dtype=np.uint8))
    return label_arrays


def multi_class_prediction(prediction, affine):
    prediction_images = []
    for i in range(prediction.shape[1]):
        prediction_images.append(get_image(prediction[0, i], affine))
    return prediction_images


def patch_wise_prediction(model, data, batch_size=1, overlap=0, permute=False):
    patch_shape = tuple([int(dim) for dim in model.input.shape[-3:]])
    indices = compute_patch_indices(data.shape[-3:], patch_size=patch_shape, overlap=overlap)
    predictions = list()
    batch = list()
    i = 0
    while i < len(indices):
        while len(batch) < batch_size:
            patch = get_patch_from_3d_data(data[0], patch_shape=patch_shape, patch_index=indices[i])
            batch.append(patch)
            i += 1
        prediction = predict(model, np.asarray(batch), permute=permute)
        batch = list()
        for predicted_patch in prediction:
            predictions.append(predicted_patch)
    output_shape = [int(model.output.shape[1])] + list(data.shape[-3:])
    return reconstruct_from_patches(predictions, patch_indices=indices, data_shape=output_shape)[np.newaxis]
