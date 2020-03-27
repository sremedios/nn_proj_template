

# TODO: format for TFRecords




import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import shutil
import time

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
import nibabel as nib
import tensorflow as tf
from tqdm import tqdm
from pathlib import Path

from utils.forward import *
from utils.progbar import *
from utils.load_ham import get_fold_fnames, fnames_to_dataset

from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

from models.densenet import *

def running_average(old_average, cur_val, n):
    return old_average * (n-1)/n + cur_val/n


if __name__ == '__main__':
    N_EPOCHS = 100000
    BATCH_SIZE = 2**6
    BUFFER_SIZE = BATCH_SIZE * 2
    learning_rate = 1e-4
    test_color_code = "\033[0;37m"
    ds = 1
    progbar_length = 5
    taus = [1.0] * 4
    TARGET_SHAPE = (112, 150, 3)
    DATA_DIR = Path("/nfs/masi/hansencb/HAM10000")

    # arguments
    experiment_mode = sys.argv[1]
    cur_fold = int(sys.argv[2])
    if experiment_mode in ["kappa", "eta", "tau", "both"]:
        weighted = True 
    elif experiment_mode == "neither":
        weighted = False
    else:
        print("Invalid weighting argument: \"{}\"".format(sys.argv[1]))
        sys.exit()
    ##### MODEL SETUP #####
    if weighted:
        weighted_str = experiment_mode + "_weighted"
    else:
        weighted_str = "unweighted"

    MODEL_NAME = "HAM_densenet169_lr_data_aug_1e-5_multitask_{}_fold_{}".format(weighted_str, cur_fold)
    
    WEIGHT_DIR = Path("models/weights") / MODEL_NAME
    RESULTS_DIR = Path("results") / MODEL_NAME
    TEST_PRED_FILENAME = RESULTS_DIR / "test_preds.csv"

    for d in [WEIGHT_DIR, RESULTS_DIR]:
        if not d.exists():
            d.mkdir(parents=True)

    model = DenseNet169_multitask(TARGET_SHAPE)


    ##### LOAD DATA #####

    targets = ["Benign-Malignant-Vascular", "Melanocytes", "Cancer", "Classify"]
    # task format: (short_name, num_outputs, long_name)
    tasks = [
        ('a', 2, targets[0]),
        ('b', 2, targets[1]),
        ('c', 2, targets[2]),
        ('d', 7, targets[3]),
    ]
    ds_model = tf.keras.models.Sequential([tf.keras.layers.AveragePooling2D(4)])

    train_fnames, val_fnames, test_fnames = get_fold_fnames(DATA_DIR, cur_fold)

    print("Loading data into RAM...")
    test_dataset = fnames_to_dataset(
            test_fnames,
            TARGET_SHAPE,
            ds_model,
            cur_fold,
            targets,
            weighted=weighted,
            batch_size=BATCH_SIZE,
            buffer_size=BUFFER_SIZE,
            training=True,
        )

    ##### EVALUATE #####
    test_loss_total = tf.keras.metrics.Mean(name='test_loss_total')
    test_loss_a = tf.keras.metrics.Mean(name='test_loss_a')
    test_accuracy_a = tf.keras.metrics.Accuracy(name='test_accuracy_a')
    test_loss_b = tf.keras.metrics.Mean(name='test_loss_b')
    test_accuracy_b = tf.keras.metrics.Accuracy(name='test_accuracy_b')
    test_loss_c = tf.keras.metrics.Mean(name='test_loss_c')
    test_accuracy_c = tf.keras.metrics.Accuracy(name='test_accuracy_c')
    test_loss_d = tf.keras.metrics.Mean(name='test_loss_d')
    test_accuracy_d = tf.keras.metrics.Accuracy(name='test_accuracy_d')

    # test metrics
    model.load_weights(
        str(WEIGHT_DIR / "best_weights.h5")
    )

    y_pred_a = []
    y_true_a = []
    y_pred_b = []
    y_true_b = []
    y_pred_c = []
    y_true_c = []
    y_pred_d = []
    y_true_d = []

    loss_descriptions = ["Loss {}".format(i) for i in ['total', 'a', 'b', 'c', 'd']]
    acc_descriptions = ["Acc {}".format(i) for i in ['total', 'a', 'b', 'c', 'd']]

    cur_step = 1
    iterator = iter(test_dataset)
    elapsed_time = 0.0
    while True:
        try:
            batch_start_time = time.time()
            data = next(iterator)
        except StopIteration:
            break
        else:
            imgs, y_a, y_b, y_c, y_d, *_ = data
            ys = [y_a, y_b, y_c, y_d]
            losses, preds = forward_multitask(
                inputs=data,
                model=model,
                loss_fns=[tf.nn.softmax_cross_entropy_with_logits]*4,
                taus=taus,
                experiment_mode=experiment_mode,
                training=False,
                num_task_specific_layers=1,
            )


            loss_total, loss_a, loss_b, loss_c, loss_d = losses
            test_loss_total.update_state(loss_total)
            test_loss_a.update_state(loss_a)
            test_loss_b.update_state(loss_b)
            test_loss_c.update_state(loss_c)
            test_loss_d.update_state(loss_d)

            pred_a, pred_b, pred_c, pred_d = preds

            test_accuracy_a.update_state(
                    y_a,
                    tf.argmax(pred_a, axis=1)
                )
            test_accuracy_b.update_state(
                    y_b,
                    tf.argmax(pred_b, axis=1)
                )
            test_accuracy_c.update_state(
                    y_c,
                    tf.argmax(pred_c, axis=1)
                )
            test_accuracy_d.update_state(
                    y_d,
                    tf.argmax(pred_d, axis=1)
                )

            y_true_a.extend(y_a)
            y_pred_a.extend(tf.argmax(pred_a, axis=1))

            y_true_b.extend(y_b)
            y_pred_b.extend(pred_b[:,1]) # keep vector element, index into "prob of class 1"

            y_true_c.extend(y_c)
            y_pred_c.extend(pred_c[:,1]) # keep vector element, index into "prob of class 1"

            y_true_d.extend(y_d)
            y_pred_d.extend(tf.argmax(pred_d, axis=1)) # threshold digit classification


            batch_end_time = time.time()
            elapsed_time = running_average(
                elapsed_time,
                batch_end_time - batch_start_time,
                cur_step,
            )
            show_progbar(
                1,
                1,
                BATCH_SIZE,
                cur_step,
                len(test_fnames),
                loss_descriptions,
                losses,
                acc_descriptions,
                [
                    test_accuracy_a.result(),
                    test_accuracy_b.result(),
                    test_accuracy_c.result(),
                    test_accuracy_d.result(),
                ],
                test_color_code,
                elapsed_time,
                progbar_length=progbar_length,
            )

            cur_step += 1

    with open(str(TEST_PRED_FILENAME), 'w') as f:
        f.write(
                (
                    "y_pred_a,y_true_a,"
                    "y_pred_b,y_true_b,"
                    "y_pred_c,y_true_c,"
                    "y_pred_d,y_true_d"
                    "\n")
        )

    with open(str(TEST_PRED_FILENAME), 'a') as f:
        for y_p_a, y_t_a, y_p_b, y_t_b, y_p_c, y_t_c, y_p_d, y_t_d in\
            zip(y_pred_a, y_true_a, y_pred_b, y_true_b, y_pred_c, y_true_c, y_pred_d, y_true_d):
            f.write(
                    (
                        "{:.4f},{:.4f},"
                        "{:.4f},{:.4f},"
                        "{:.4f},{:.4f},"
                        "{:.4f},{:.4f}"
                        "\n").format(
                y_p_a.numpy(),
                y_t_a.numpy(),
                y_p_b.numpy(),
                y_t_b.numpy(),
                y_p_c.numpy(),
                y_t_c.numpy(),
                y_p_d.numpy(),
                y_t_d.numpy(),
            ))
