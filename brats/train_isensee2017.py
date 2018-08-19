import os

from unet3d.generator import get_training_and_validation_generators
from unet3d.model import isensee2017_model
from unet3d.training import load_old_model, train_model
from brats.preprocess2015 import pre_config


config = dict()
config["image_shape"] = pre_config["image_shape"]  # This determines what shape the images will be cropped/resampled to.
config["patch_shape"] = None  # switch to None to train on the whole image
config["labels"] = (1, 2, 3, 4)  # the label numbers on the input image
config["n_base_filters"] = 16
config["n_labels"] = len(config["labels"])
config["all_modalities"] = ["t1", "t1Gd", "flair", "t2"]
config["training_modalities"] = config["all_modalities"]  # change this if you want to only use some of the modalities
config["nb_channels"] = len(config["training_modalities"])
if "patch_shape" in config and config["patch_shape"] is not None:
    config["input_shape"] = tuple([config["nb_channels"]] + list(config["patch_shape"]))
else:
    config["input_shape"] = tuple([config["nb_channels"]] + list(config["image_shape"]))

config["batch_size"] = 1
config["validation_batch_size"] = 1
config["n_epochs"] = 500  # cutoff the training after this many epochs
config["patience"] = 10  # learning rate will be reduced after this many epochs if the validation loss is not improving
config["early_stop"] = 50  # training will be stopped after this many epochs without the validation loss improving
config["initial_learning_rate"] = 5e-4
config["learning_rate_drop"] = 0.5  # factor by which the learning rate will be reduced
config["validation_split"] = 0.8  # portion of the data that will be used for training
config["flip"] = False  # augments the data by randomly flipping an axis during
config["permute"] = True  # data shape must be a cube. Augments the data by permuting in various directions
config["distort"] = None  # switch to None if you want no distortion
config["augment"] = config["flip"] or config["distort"]
config["validation_patch_overlap"] = 0  # if > 0, during training, validation patches will be overlapping
config["training_patch_start_offset"] = (16, 16, 16)  # randomly offset the first patch index by up to this offset
config["skip_blank"] = True  # if True, then patches without any target will be skipped

config["overwrite"] = False  # If True, will previous files. If False, will use previously written files.
config["model_file"] = os.path.abspath("isensee_2017_model.h5")
config["training_keys_file"] = os.path.abspath("isensee_training_ids.pkl")
config["validation_keys_file"] = os.path.abspath("isensee_validation_ids.pkl")
config["subject_ids_file"] = pre_config["subject_ids_file"]
config["npy_path"] = pre_config["npy_path"]


def main(overwrite=False):
    if not overwrite and os.path.exists(config["model_file"]):
        model = load_old_model(config["model_file"])
    else:
        # instantiate new model
        model = isensee2017_model(input_shape=config["input_shape"],
                                  n_labels=config["n_labels"],
                                  initial_learning_rate=config["initial_learning_rate"],
                                  n_base_filters=config["n_base_filters"])

    # get training and validation generators
    train_generator, validation_generator, n_train_steps, n_validation_steps = get_training_and_validation_generators(
        npy_path=config["npy_path"],
        subject_ids_file=config["subject_ids_file"],
        batch_size=config["batch_size"],
        validation_batch_size=config["validation_batch_size"],
        n_labels=config["n_labels"],
        labels=config["labels"],

        training_keys_file=config["training_keys_file"],
        validation_keys_file=config["validation_keys_file"],
        data_split=config["validation_split"],
        overwrite=overwrite,

        augment=config["augment"],
        augment_flip=config["flip"],
        augment_distortion_factor=config["distort"],
        permute=config["permute"],

        image_shape=config["image_shape"],
        patch_shape=config["patch_shape"],

        validation_patch_overlap=config["validation_patch_overlap"],
        training_patch_start_offset=config["training_patch_start_offset"],

        skip_blank=config["skip_blank"]
        )

    # run training
    train_model(model=model,
                model_file=config["model_file"],
                training_generator=train_generator,
                validation_generator=validation_generator,
                steps_per_epoch=n_train_steps,
                validation_steps=n_validation_steps,
                initial_learning_rate=config["initial_learning_rate"],
                learning_rate_drop=config["learning_rate_drop"],
                learning_rate_patience=config["patience"],
                early_stopping_patience=config["early_stop"],
                n_epochs=config["n_epochs"])


if __name__ == "__main__":
    main(overwrite=config["overwrite"])
