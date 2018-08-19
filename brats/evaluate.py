import os
import glob
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def get_whole_tumor_mask(data):
    return data > 0


def get_tumor_core_mask(data):
    return np.logical_or(data == 1, data == 4)


def get_enhancing_tumor_mask(data):
    return data == 4


def dice_coefficient(truth, prediction, smooth=0.00001):
    return 2 * np.sum(truth * prediction + smooth / 2) / (np.sum(truth) + np.sum(prediction) + smooth)


def main():
    header = ("WholeTumor", "TumorCore", "EnhancingTumor")
    masking_functions = (get_whole_tumor_mask, get_tumor_core_mask, get_enhancing_tumor_mask)
    rows = list()
    subject_ids = list()
    input_folder = "data/prediction"
    for subject_folder in glob.glob(os.path.join(input_folder, "*", "*")):
        if not os.path.isdir(subject_folder):
            continue
        print('find subject', subject_folder)
        subject_id = os.path.basename(subject_folder)
        subject_category = os.path.basename(os.path.dirname(subject_folder))
        subject_ids.append(os.path.join(subject_category, subject_id))

        truth_file = os.path.join(subject_folder, "truth.nii.gz")
        truth_image = nib.load(truth_file)
        truth = truth_image.get_data()
        prediction_file = os.path.join(subject_folder, "prediction.nii.gz")
        prediction_image = nib.load(prediction_file)
        prediction = prediction_image.get_data()
        rows.append([dice_coefficient(func(truth), func(prediction))for func in masking_functions])

    df = pd.DataFrame.from_records(rows, columns=header, index=subject_ids)
    df.to_csv("data/prediction/brats_scores.csv")

    scores = dict()
    for index, score in enumerate(df.columns):
        values = df.values.T[index]
        scores[score] = values[np.isnan(values) == False]

    plt.boxplot(list(scores.values()), labels=list(scores.keys()))
    plt.ylabel("Dice Coefficient")
    plt.savefig("data/prediction/validation_scores_boxplot.png")
    plt.close()

    if os.path.exists("./training.log"):
        training_df = pd.read_csv("./training.log").set_index('epoch')

        plt.plot(training_df['loss'].values, label='training loss')
        plt.plot(training_df['val_loss'].values, label='validation loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.xlim((0, len(training_df.index)))
        plt.legend(loc='upper right')
        plt.savefig('data/prediction/loss_graph.png')


if __name__ == "__main__":
    main()
