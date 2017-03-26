#!/usr/bin/env python3

from glob import glob
import base64
import csv
import functools
import hashlib
import itertools
import json
import logging
import logging.handlers
import math
import os
import os.path
import pickle
import pprint
import random
import time

from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.layers import Activation, merge
from keras.layers import Input, Flatten, Dense, Lambda, Cropping2D, Dropout, ELU
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU, SReLU
from keras.layers.convolutional import Convolution2D
from keras.layers.noise import GaussianNoise
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
import cv2
import hyperopt
import hyperopt.hp
import numpy as np
import sklearn
import sklearn.model_selection
import sklearn.utils
import tensorflow as tf

from util import preprocess_img, augment_batch, translation


logger = logging.getLogger('model')
logger.setLevel(logging.DEBUG)
fh = logging.handlers.RotatingFileHandler('output.log', maxBytes=10*1024*1024, backupCount=2)
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)


def get_driving_log_path(image_filepath, path):
    elems = [path] + image_filepath.split(os.sep)[-3:]
    elems = [elem.strip() for elem in elems]
    return os.path.abspath(os.sep.join(elems))


def load_driving_log(path, filepath):
    fieldnames = [
        'center',
        'left',
        'right',
        'steering',
        'throttle',
        'brake',
        'speed',
    ]
    output = []
    with open(filepath) as f_in:
        first_line = f_in.readline()
        has_headers = 'steering' in first_line
    with open(filepath) as f_in:
        if has_headers:
            reader = csv.DictReader(f_in)
        else:
            reader = csv.DictReader(f_in, fieldnames=fieldnames)
        for row in reader:
            row['center'] = get_driving_log_path(row['center'], path)
            row['left'] = get_driving_log_path(row['left'], path)
            row['right'] = get_driving_log_path(row['right'], path)
            row['steering'] = float(row['steering'])
            row['throttle'] = float(row['throttle'])
            row['brake'] = float(row['brake'])
            row['speed'] = float(row['speed'])
            output.append(row)
    return output


def get_all_driving_logs(path='training_data_hq'):
    driving_logs = glob(os.path.join(path, "*", "driving_log.csv"))
    logger.info("driving_logs paths: %s" % (driving_logs, ))
    driving_logs = [load_driving_log(path, log) for log in driving_logs]
    logs = [log for driving_log in driving_logs for log in driving_log]
    assert(all(os.path.isfile(log['center']) for log in logs))
    assert(all(os.path.isfile(log['left']) for log in logs))
    assert(all(os.path.isfile(log['right']) for log in logs))
    return logs


def get_flatter_sample(logs, number_of_bins=25, max_factor=5):
    angles = [abs(log['steering']) for log in logs]
    average_samples_per_bin = math.ceil(len(angles) / number_of_bins)
    hist, bins = np.histogram(angles, number_of_bins)
    
    new_logs = []
    for bin_start, bin_end in zip(bins, bins[1:]):
        subset = [log for log in logs
                  if abs(log['steering']) >= bin_start and
                     abs(log['steering']) <= bin_end]
        if len(subset) >= average_samples_per_bin:
            size = max(average_samples_per_bin, math.ceil(len(subset) / max_factor))
        else:
            size = min(average_samples_per_bin, math.ceil(len(subset) * max_factor))
        replace = size >= len(subset)
        sampled = np.random.choice(subset,
                                   size=size,
                                   replace=replace)
        new_logs.extend(sampled)
    return sklearn.utils.shuffle(new_logs)


def opencv_read(filepath):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def generator(samples,
              is_training,
              crop_top=0,
              batch_size=32,
              steering_delta=0.2,
              translation_delta=0.005,
              number_of_bins=25,
              augmentation_new_image_count=10):
    samples = sorted(samples, key=lambda x: x['center'])
    num_samples = len(samples)
    while True:
        current_samples = get_flatter_sample(samples,
                                             number_of_bins=number_of_bins)
        for offset in range(0, num_samples, batch_size):
            batch_samples = current_samples[offset:offset+batch_size]
            if len(batch_samples) == 0:
                continue
            center_images = [opencv_read(sample['center']) for sample in batch_samples]
            left_images = [opencv_read(sample['left']) for sample in batch_samples]
            right_images = [opencv_read(sample['right']) for sample in batch_samples]
            center_steerings = [sample['steering'] for sample in batch_samples]
            left_steerings = [sample['steering'] + steering_delta for sample in batch_samples]
            right_steerings = [sample['steering'] - steering_delta for sample in batch_samples]
            images_steerings = list(zip(center_images, center_steerings))
            if is_training:
                flipped_images_steerings = images_steerings + \
                    [(cv2.flip(img, 1), -steering)
                     for (img, steering) in images_steerings]

                images_steerings = images_steerings + flipped_images_steerings
                images_steerings = images_steerings + list(zip(left_images, left_steerings))
                images_steerings = images_steerings + list(zip(right_images, right_steerings))

                translated_images_steerings = []
                for (img, steering) in images_steerings:
                    new_img, t_x = translation(img)
                    new_steering = steering + translation_delta * t_x
                    translated_images_steerings.append((new_img, new_steering))

                images_steerings = images_steerings + translated_images_steerings
                
            random_images_steerings = random.sample(images_steerings, len(batch_samples))
            images, steerings = zip(*random_images_steerings)

            X_train = images
            y_train = steerings
            if is_training:
                X_train, y_train = augment_batch(
                    X_train,
                    y_train,
                    new_image_count=augmentation_new_image_count,
                    with_contrast=True,
                    with_shadow=True,
                    with_shear=True)
            X_train, y_train = sklearn.utils.shuffle(X_train, y_train)
            X_train = np.array([preprocess_img(img, crop_top) for img in X_train])
            yield (X_train, y_train)


def get_conv_activation(x, name):
    if name == 'relu':
        return Activation('relu')(x)
    elif name == 'elu':
        return ELU()(x)
    elif name == 'prelu':
        return PReLU()(x)
    elif name == 'srelu':
        return SReLU()(x)
    elif name == 'none':
        return x


def get_model(channels,
              input_rows,
              input_cols,
              conv_activation,
              conv_dropout,
              flatten_dropout,
              flatten_activation,
              dense_activation,
              dense_dropout,
              use_initial_scaling,
              conv_filters,
              conv_kernels,
              max_pools,
              fc_depths):
    img_input = Input(shape=(input_rows, input_cols, channels))

    if use_initial_scaling:
        x = Convolution2D(3, 1, 1)(img_input)
        x = get_conv_activation(x, conv_activation)
    else:
        x = img_input

    for i in range(len(conv_filters)):
        x = Convolution2D(conv_filters[i], conv_kernels[i], conv_kernels[i])(x)
        x = get_conv_activation(x, conv_activation)
        x = MaxPooling2D((max_pools[i], max_pools[i]))(x)
        x = Dropout(conv_dropout)(x)

    x = Flatten(name='flatten')(x)
    x = Dropout(flatten_dropout)(x)
    x = get_conv_activation(x, flatten_activation)

    y = x
    for i, fc_depth in enumerate(fc_depths, start=1):
        y = Dense(fc_depth, name='fc%s_steering' % (i, ))(y)
        y = Dropout(dense_dropout)(y)
        y = get_conv_activation(y, dense_activation)
    steering = Dense(1, name='steering')(y)

    return Model(
        input=[img_input],
        output=[steering],
    )


def train_model(kwargs, print_model_summary=False):
    epochs = 100
    batch_size = 50
    number_of_bins = 25
    augmentation_new_image_count = 4
    learning_rate = 2e-4  # default is 1e-3

    logger.info(pprint.pformat(kwargs))
    conv_activation = 'prelu'
    use_initial_scaling = False
    conv_dropout = kwargs['conv_dropout']
    crop_top = int(kwargs['crop_top'])
    steering_delta = kwargs['steering_delta']
    translation_delta = kwargs['translation_delta']
    flatten_dropout =  kwargs['flatten_dropout']
    flatten_activation = kwargs['flatten_activation']
    dense_activation = kwargs['dense_activation']
    dense_dropout = kwargs['dense_dropout']
    conv_filters = list(map(int, kwargs['conv_filters']))
    conv_kernels = list(map(int, kwargs['conv_kernels']))
    fc_depths = list(map(int, kwargs['fc_depths']))
    max_pools = list(map(int, kwargs['max_pools']))

    logs = get_all_driving_logs(path='training_data_hq')
    train, valid = sklearn.model_selection.train_test_split(logs, test_size=0.1, random_state=1)
    sample_generator = functools.partial(generator,
        crop_top=crop_top,
        batch_size=batch_size,
        steering_delta=steering_delta,
        translation_delta=translation_delta,
        number_of_bins=number_of_bins,
        augmentation_new_image_count=augmentation_new_image_count)
    train_generator = sample_generator(train, is_training=True)
    valid_generator = sample_generator(valid, is_training=False)

    try:
        model = get_model(
            channels=3,
            input_rows=128,
            input_cols=128,
            conv_activation=conv_activation,
            conv_dropout=conv_dropout,
            flatten_dropout=flatten_dropout,
            flatten_activation=flatten_activation,
            dense_activation=dense_activation,
            dense_dropout=dense_dropout,
            use_initial_scaling=use_initial_scaling,
            conv_filters=conv_filters,
            conv_kernels=conv_kernels,
            fc_depths=fc_depths,
            max_pools=max_pools)
        if print_model_summary:
            model.summary()
        adam = Adam(lr=learning_rate)
        model.compile(optimizer=adam, loss={
            'steering': 'mean_squared_error',
        })

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=2)
        key = hashlib.md5(bytes(json.dumps(kwargs), 'utf-8')).hexdigest()
        model_checkpoint = ModelCheckpoint(
            'model-{val_loss:.4f}-%s-{epoch:02d}.h5' % (key, ),
            monitor='val_loss',
            verbose=0,
            save_best_only=False)
        tensorboard = TensorBoard(
            log_dir='./tensorboard',
            histogram_freq=0,
            write_graph=True,
            write_images=True)

        history = model.fit_generator(
            train_generator,
            samples_per_epoch=len(train)*augmentation_new_image_count,
            validation_data=valid_generator,
            nb_val_samples=len(valid),
            callbacks=[
                model_checkpoint,
                early_stopping,
                #tensorboard,
            ],
            nb_epoch=epochs,
            verbose=1)
    except Exception as e:
        logger.exception("Failed training parameters %s" % (pprint.pformat(kwargs), ))
        if K.backend() == 'tensorflow':
            K.clear_session()
        return {
            'status': hyperopt.STATUS_FAIL,
            'time': time.time(),
            'exception': str(e),
        }

    val_loss = history.history['val_loss']
    loss = history.history['loss']
    if K.backend() == 'tensorflow':
        K.clear_session()
    logger.info("%.4f val_loss for model key %s: %s" % (min(val_loss), key, pprint.pformat(kwargs)))
    return {
        'loss': min(val_loss),
        'time': time.time(),
        'status': hyperopt.STATUS_OK,
    }



def find_best_model():
    space = hyperopt.hp.choice('parameters', [{
        # between 20 to 70 is good, 55 seems best
        'crop_top': hyperopt.hp.quniform('crop_top',
            0, 70, 5),

        # some value between 0.2 and 0.275 is best, 0.225 seems best
        'steering_delta': hyperopt.hp.quniform('steering_delta',
            0.2, 0.3, 0.025),

        # 0.007 seems best
        'translation_delta': hyperopt.hp.quniform('translation_delta',
            0.004, 0.008, 0.001),

        # exclude srelu because it doubles training time
        # prelu is consistently chosen, so use that.
        #'conv_activation': hyperopt.hp.choice('conv_activation',
        #    ['relu', 'elu', 'prelu']),

        'flatten_dropout': hyperopt.hp.quniform('flatten_dropout',
            0.0, 1.0, 0.1),

        'flatten_activation': hyperopt.hp.choice('flatten_activation',
            ['prelu']),

        'dense_dropout': hyperopt.hp.quniform('dense_dropout',
            0.0, 1.0, 0.1),

        'dense_activation': hyperopt.hp.choice('dense_activation',
            ['prelu']),

        'conv_dropout': hyperopt.hp.quniform('conv_dropout',
            0.0, 1.0, 0.1),

        # consistently gets set to True, helpful
        #'use_initial_scaling': hyperopt.hp.choice('use_initial_scaling',
        #    [True, False]),

        # does a second or third layer help? it could be that intermediate
        # fc layers do some feature extraction as well as the convnets.
        'fc_depths': hyperopt.hp.choice('fc_depths', [
            [512],
            [512, 512],
            [512, 512, 512],
        ]),
        
        # settle on 64, 96, 128, 160
        'conv_filters': hyperopt.hp.choice('conv_filters', [
            #[16, 32, 64, 128],
            [64, 96, 128, 160],
        ]),

        # settle on 7, 5, 3, 3
        'conv_kernels': [
            hyperopt.hp.choice('conv_kernels[0]', [7]),
            hyperopt.hp.choice('conv_kernels[1]', [5]),
            hyperopt.hp.choice('conv_kernels[2]', [3]),
            hyperopt.hp.choice('conv_kernels[3]', [3]),
        ],

        # settle on 3, 2, 2, 2
        'max_pools': [
            hyperopt.hp.choice('max_pools[0]', [3]),
            hyperopt.hp.choice('max_pools[1]', [2]),
            hyperopt.hp.choice('max_pools[2]', [2]),
            hyperopt.hp.choice('max_pools[3]', [2]),
        ],
    }])
    # hyperopt.pyll.stochastic.sample(space)
    trials_path = 'trials.pickle'
    if os.path.isfile(trials_path):
        with open(trials_path, 'rb') as f_in:
            trials = pickle.load(f_in)
    else:
        trials = hyperopt.Trials()
    best = hyperopt.fmin(train_model,
        space=space,
        algo=functools.partial(hyperopt.tpe.suggest, n_startup_jobs=3),
        max_evals=len(trials.trials)+10,
        trials=trials,
        verbose=1,
        rstate=np.random.RandomState(2))
    logger.info(best)
    logger.info(trials.best_trial)
    with open(trials_path, 'wb') as f_out:
        pickle.dump(trials, f_out)


def main():
    #while True:
    #    find_best_model()

    train_model({
        'conv_dropout': 0.1,
        'conv_filters': [64, 96, 128, 160],
        'conv_kernels': [7, 5, 3, 3],
        'crop_top': 30,
        'dense_activation': 'prelu',
        'dense_dropout': 0.2,
        'fc_depths': [512],
        'flatten_activation': 'prelu',
        'flatten_dropout': 0.2,
        'max_pools': [3, 2, 2, 2],
        'steering_delta': 0.250,
        'translation_delta': 0.007,
    }, print_model_summary=True)

if __name__ == '__main__':
    main()
