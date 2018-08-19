import os

from unet3d.prediction import run_validation_cases
from brats.preprocess2015 import pre_config


def main(model_name='isensee2017'):
    if model_name == 'unet':
        from brats.train import config
    else:
        from brats.train_isensee2017 import config

    prediction_dir = os.path.abspath(os.path.join("data", "prediction"))
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir)
    run_validation_cases(model_file=config["model_file"],
                         training_modalities=config["training_modalities"],
                         validation_keys_file=config["validation_keys_file"],
                         npy_path=pre_config["npy_path"],
                         subject_ids_file=pre_config["subject_ids_file"],
                         label_map=True,
                         output_dir=prediction_dir,
                         labels=config["labels"])


if __name__ == "__main__":
    main()
