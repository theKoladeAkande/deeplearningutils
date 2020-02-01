#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import defaultdict
import time
import os
import numpy as np
from tensorflow import keras
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import save_model
import tensorflow as tf
import matplotlib.pyplot as plt
import tqdm_utils


class TqdmProgressCallback(Callback):

    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']


    def on_epoch_begin(self, epoch, logs=None):
        print('\nEpoch %d/%d' % (epoch + 1, self.epochs))
        if "steps" in self.params:
            self.use_steps = True
            self.target = self.params['steps']
        else:
            self.use_steps = False
            self.target = self.params['samples']
        self.prog_bar = tqdm_utils.tqdm_notebook_failsafe(total=self.target)
        self.log_values_by_metric = defaultdict(list)

    def _set_prog_bar_desc(self, logs):
        for k in self.params['metrics']:
            if k in logs:
                self.log_values_by_metric[k].append(logs[k])
        desc = "; ".join("{0}: {1:.4f}".format(k, np.mean(values)) for k, values in self.log_values_by_metric.items())
        if hasattr(self.prog_bar, "set_description_str"):  # for new tqdm versions
            self.prog_bar.set_description_str(desc)
        else:
            self.prog_bar.set_description(desc)

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        if self.use_steps:
            self.prog_bar.update(1)
        else:
            batch_size = logs.get('size', 0)
            self.prog_bar.update(batch_size)
        self._set_prog_bar_desc(logs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self._set_prog_bar_desc(logs)
        self.prog_bar.update(1)  # workaround to show description
        self.prog_bar.close()
        print("\nval/train: {:.2f}".format(logs["val_loss"] / logs["loss"]))



class ModelSaveCallback(Callback):

    def __init__(self, file_name):
        super(ModelSaveCallback, self).__init__()
        self.file_name = file_name

    def on_epoch_end(self, epoch, logs=None):
        model_filename = self.file_name.format(epoch)
        save_model(self.model, model_filename)
        print("Model saved in {}".format(model_filename))


def get_run_logdir(f_name='my_logs'): # create log directory for Tensorboard
    "f_name: folder name"
    cwd_fpath = os.path.join(os.getenv('HOME'),f_name)
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(cwd_fpath, run_id)


def display_random_images(x_train,y_train,classes_tag,n_cols=8, n_rows=2):
    
    fig = plt.figure(figsize=(2 * n_cols - 1, 2.5 * n_rows - 1))
    for i in range(n_cols):
        for j in range(n_rows):
            random_index = np.random.randint(0, len(y_train))
            nindex = (i * n_rows) + (j + 1)
            ax = fig.add_subplot(n_rows, n_cols,nindex)
            ax.grid('off')
            ax.axis('off')
            ax.imshow(x_train[random_index, :])
            ax.set_title(classes_tag[y_train[random_index, 0]])
    plt.show()
