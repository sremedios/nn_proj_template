import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import shutil
import time

import numpy as np
import nibabel as nib
import tensorflow as tf
from tqdm import tqdm
from pathlib import Path

from utils.load_ham import *
from utils.tfrecord_utils import rgb_example

from sklearn.utils import shuffle


if __name__ == '__main__':

    # arguments
    IN_DATA_DIR = Path(sys.argv[1])
    OUT_DATA_DIR = Path(sys.argv[2])
    cur_fold = int(sys.argv[3])

    if not OUT_DATA_DIR.exists():
        OUT_DATA_DIR.mkdir(parents=True)

    TFRECORD_FNAME = OUT_DATA_DIR / "dataset_fold_{}_{}.tfrecord"
    TARGET_SHAPE = (224, 224)

    count_file = OUT_DATA_DIR / "data_count.txt"

    ##### LOAD DATA #####

    targets = ["Benign-Malignant-Vascular", "Melanocytes", "Cancer", "Classify"]

    print("Splitting data...")
    train_fnames, val_fnames, test_fnames = get_fold_fnames(IN_DATA_DIR, cur_fold)

    for fnames, cur_name in zip(
            [train_fnames, val_fnames, test_fnames],
            ["train", "val", "test"],
        ):

        ys = np.array([int(f.parent.name) for f in fnames])
        print("Mapping tasks...")
        ys = map_multitask(ys)
        print("Generating dataframe...")
        ys_df = make_label_df(ys, targets)
        print("Applying KDE...")
        apply_kde(ys_df, targets)

        print("Writing TFRecord...")
        counter = 0
        with tf.io.TFRecordWriter(str(TFRECORD_FNAME).format(cur_fold, cur_name)) as writer:
            for i, (fname, row) in tqdm(enumerate(zip(fnames,ys_df.iterrows())), total=len(fnames)): 
                x = load_preprocess_fname(fname, TARGET_SHAPE)
                y = np.array(row[1], dtype=np.float32)

                tf_example = rgb_example(x, y)
                writer.write(tf_example.SerializeToString())
                counter += 1

        with open(count_file, 'a') as f:
            f.write("{} {} {}\n".format(cur_name, cur_fold, counter))
