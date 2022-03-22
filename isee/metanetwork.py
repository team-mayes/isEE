"""
The purpose of this script is to test out my idea for a so-called "metanetwork" for making predictions from
simulations data. The core idea is that there are multiple individual subnetworks with distinct training data that are
connected to one another and trained on different training data asynchronously (with the ones not being
trained at the moment having fixed weights).
"""

import os
import sys
import shutil
import copy
import math
import dill as pickle
import mdtraj
import argparse
import numpy as np
import scipy.stats
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'   # force keras to use CPU (this has to be done before importing tf)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import *
from matplotlib import pyplot as plt
import keras_tuner
# from allennlp.commands.elmo import ElmoEmbedder
# from pathlib import Path
#
# model_dir = Path('C:\Users\User\PycharmProjects\isEE\uniref50_v2')
# weights = model_dir / 'weights.hdf5'
# options = model_dir / 'options.json'
# embedder = ElmoEmbedder(options, weights, cuda_device=0) # cuda_device=-1 for CPU

def get_train_and_test_splits(history, settings):
    """
    Return train and test splits from 'history' for the training data corresponding to settings.obj using the random
    seed settings.seed
    """

    muts = []
    scores = []
    wt_score = []
    for ii in range(len(history.score)):
        if history.score[ii]:
            if 'score' in settings.obj:
                scores.append(history.score[ii].rmsf_ts)
            elif settings.obj == 'rmsf':
                scores.append([item[1] for item in history.score[ii].rmsf])
            elif settings.obj == 'covar':
                scores.append(list(history.score[ii].covar))
            elif settings.obj == 'hbonds':
                scores.append(list(history.score[ii].num_hbonds)[:441])
            elif settings.obj == 'nematic':
                scores.append(history.score[ii].nematic)
            else:
                raise RuntimeError('unrecognized model_settings.obj: ' + settings.obj)
            muts.append(history.muts[ii])  # only get muts with corresponding scores
        if history.score[ii] and history.muts[ii] == ['WT']:
            if 'score' in settings.obj:
                wt_score.append(history.score[ii].rmsf_ts)
            elif settings.obj == 'rmsf':
                wt_score.append([item[1] for item in history.score[ii].rmsf])
            elif settings.obj == 'covar':
                wt_score.append(list(history.score[ii].covar))
            elif settings.obj == 'hbonds':
                wt_score.append(list(history.score[ii].num_hbonds)[:441])
            elif settings.obj == 'nematic':
                wt_score.append(history.score[ii].nematic)

    if wt_score and settings.obj in ['covar', 'rmsf', 'hbonds']:
        wt_score = [np.mean([score[ii] for score in wt_score]) for ii in range(len(wt_score[0]))]

    print('number of mutants: ' + str(len(muts)))

    # Convert each mutants list entry into full sequence-length list of integer-encoded features
    feats = []  # initialize feature vector
    for mut in muts:
        feats.append(integer_encoder(mut, settings.seq))

    # Subdivide into training and testing datasets based on train_size argument
    rng = np.random.default_rng(settings.seed)
    indices = [ind for ind in range(len(feats))]
    rng.shuffle(indices)
    test_split = (len(feats) - settings.train_size) / len(feats)

    # A kludge for now that allows me to specify mutants to withhold for the test set
    pre_indices = copy.copy(indices)
    # to_withhold = [['64ASP', '386SER', '394ALA'], ['64ASP', '386SER'], ['64ASP', '394ALA', '407PRO', '423PRO'],
    #               ['64ASP', '386SER', '360VAL', '389ASN'], ['64ASP', '386SER', '422THR']]
    train_history = pickle.load(open('data/210922_resampled_history.pkl', 'rb'))
    to_withhold = []
    for mut in muts:
        if mut not in train_history.muts:
            to_withhold.append(mut)
    indices_of_withholding = [muts.index(mut) for mut in to_withhold]  # order of single muts within each mut matters for this implementation
    print(indices_of_withholding)
    # for index in indices_of_withholding:
        # print(muts[index])
    to_exchange = []
    for ii in indices_of_withholding:
        if indices.index(ii) < int(len(feats) * (1 - test_split)):  # if ii is in the train split
            to_exchange.append(ii)
    offset = 1
    for jj in range(len(to_exchange)):
        while indices[-1 * (jj + offset)] in indices_of_withholding:
            offset += 1     # to skip swapping out members of to_withhold back into the train set
        temp = indices[-1 * (jj + offset)]
        original_index = indices.index(to_exchange[jj])
        indices[-1 * (jj + offset)] = to_exchange[jj]
        indices[original_index] = temp
        print('moving mutant at index ' + str(to_exchange[jj]) + ' (' + str(muts[to_exchange[jj]]) + ') to index: ' + str(-1 * (jj + offset)))

    muts = [muts[ii] for ii in indices]
    feats = [feats[ind] for ind in indices]
    scores = [scores[ind] for ind in indices]

    # Just a couple sanity checks to make sure withholding didn't break anything
    assert (all([item in indices for item in pre_indices]))
    assert (all([mut in muts[int(len(feats) * (1 - test_split)):] for mut in to_withhold]))

    # Normalize on train split only
    train_scores = scores[:int(len(feats) * (1 - test_split))]
    if not settings.obj in ['covar', 'rmsf', 'hbonds']:
        scores = [(score - min(train_scores)) / (max(train_scores) - min(train_scores)) for score in scores]
    else:
        normscores = []
        for score in scores:
            assert len(score) == len(scores[0])
            assert not max(score) == min(score)
            scor = [score[ii] - wt_score[ii] for ii in range(len(score))]
            train_scor = scor[:int(len(feats) * (1 - test_split))]
            normscores.append([(sco - min(train_scor)) / (max(train_scor) - min(train_scor)) for sco in scor])
        scores = normscores.copy()

    feats_train = np.array(feats[:int(len(feats) * (1 - test_split))])
    scores_train = np.array(scores[:int(len(feats) * (1 - test_split))])
    feats_test = np.array(feats[int(len(feats) * (1 - test_split)):])
    scores_test = np.array(scores[int(len(feats) * (1 - test_split)):])

    train_dataset = tf.data.Dataset.from_tensor_slices((feats_train, scores_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((feats_test, scores_test))

    return train_dataset, test_dataset


def integer_encoder(muts, wt_seq):
    # Encode sequence with mutations as a list of integers

    all_resnames = ['ARG', 'HIS', 'LYS', 'ASP', 'GLU', 'SER', 'THR', 'ASN', 'GLN', 'CYS', 'GLY', 'PRO', 'ALA',
                    'VAL', 'ILE', 'LEU', 'MET', 'PHE', 'TYR', 'TRP']  # all resnames in encoding order

    if muts == ['WT']:  # special case
        muts = []  # no mutation applied

    values = copy.copy(wt_seq)
    for mut in muts:
        values[int(mut[:-3]) - 1] = mut[-3:]

    return [all_resnames.index(item) for item in values]


def run_experiment(model, train_dataset, test_dataset, num_epochs, settings):
    # tf.keras.mixed_precision.experimental.set_policy('mixed_float16')
    # tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
    # tf.keras.mixed_precision.set_global_policy("mixed_float16")
    # os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

    # from tensorflow.python.framework.ops import disable_eager_execution
    # disable_eager_execution()

    tf.config.optimizer.set_jit(True)

    # tb_callback = tf.keras.callbacks.TensorBoard(log_dir='logs', profile_batch='10, 20')
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=500, monitor='val_mean_squared_error', restore_best_weights=True, verbose=1)

    # tuner = keras_tuner.tuners.Hyperband(
    #     create_tunable_model,
    #     objective='val_mean_squared_error',
    #     max_epochs=500,
    #     directory='hyperband_score')
    # )

    tuner = keras_tuner.tuners.BayesianOptimization(
        create_tunable_model,
        objective='val_mean_squared_error',
        max_trials=1000
    )

    tuner.search(train_dataset, validation_data=test_dataset)
    tuner.results_summary()

    sys.exit()

    print("Start training the model...")
    history = model.fit(train_dataset.prefetch(settings.batch_size * 10), epochs=num_epochs, validation_data=test_dataset, use_multiprocessing=True,
                        workers=8)#, callbacks=[early_stopping])
    print("Model training finished.")
    _, rmse = model.evaluate(train_dataset, verbose=0)
    print(f"Train RMSE: {round(rmse, 4)}")

    print("Evaluating model performance...")
    _, rmse = model.evaluate(test_dataset, verbose=0)
    print(f"Test RMSE: {round(rmse, 4)}")

    return history


def create_tunable_model(hp):
    initializer = tf.keras.initializers.HeUniform()

    settings = argparse.Namespace()
    settings.batch_size = 19
    settings.obj = 'full_score'

    input = Input(shape=(441), name='input')

    # Direct score subnetwork
    score_backbone = tf.keras.Sequential([
        tf.keras.layers.Embedding(20, hp.Int('encoding', min_value=2, max_value=42, step=4), input_length=441),
        tf.keras.layers.Conv1D(hp.Int('conv1_filters', min_value=8, max_value=80, step=8), hp.Int('conv1_width', min_value=2, max_value=29, step=3), hp.Int('conv1_stride', min_value=1, max_value=3, step=1), activation=keras.layers.LeakyReLU(), kernel_initializer=initializer),
        tf.keras.layers.Conv1D(hp.Int('conv2_filters', min_value=8, max_value=80, step=8), hp.Int('conv2_width', min_value=2, max_value=29, step=3), hp.Int('conv2_stride', min_value=1, max_value=3, step=1), activation=keras.layers.LeakyReLU(), kernel_initializer=initializer),
        tf.keras.layers.Conv1D(hp.Int('conv3_filters', min_value=8, max_value=80, step=8), hp.Int('conv3_width', min_value=2, max_value=29, step=3), hp.Int('conv3_stride', min_value=1, max_value=3, step=1), activation=keras.layers.LeakyReLU(), kernel_initializer=initializer),
        tf.keras.layers.GlobalAveragePooling1D(),
        # tf.keras.layers.Dense(16, activation='relu', kernel_initializer=initializer),
    ], name='score_backbone')
    score_head = tf.keras.Sequential([
        tf.keras.layers.Dense(hp.Int('dense1', min_value=10, max_value=200, step=10), activation=keras.layers.LeakyReLU(), kernel_initializer=initializer),
        tf.keras.layers.Dropout(hp.Int('dropout', min_value=1, max_value=5, step=1)/10),
        # tf.keras.layers.Dense(16, activation=keras.layers.LeakyReLU(), kernel_initializer=initializer),
        tf.keras.layers.Dense(1)
    ], name='score_head')

    # RMSF subnetwork
    rmsf_backbone = tf.keras.Sequential([
        tf.keras.layers.Embedding(20, 6, input_length=441),
        tf.keras.layers.Conv1D(8, 8, 2, activation='relu', kernel_initializer=initializer),
        tf.keras.layers.Conv1D(104, 6, 1, activation='relu', kernel_initializer=initializer),
        tf.keras.layers.Conv1D(40, 4, 3, activation='relu', kernel_initializer=initializer),
        tf.keras.layers.GlobalAveragePooling1D(),
        # tf.keras.layers.Dense(16, activation='relu', kernel_initializer=initializer),
    ], name='rmsf_backbone')
    rmsf_head = tf.keras.Sequential([
        tf.keras.layers.Dense(12, activation='relu', kernel_initializer=initializer),
        tf.keras.layers.Dense(32, activation='relu', kernel_initializer=initializer),
        tf.keras.layers.Dense(440, activation='linear', dtype=tf.float32)
    ], name='rmsf_head')
    rmsf_residual = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', kernel_initializer=initializer)  # 16 best
    ], name='rmsf_residual')
    rmsf_score = tf.keras.Sequential([
        tf.keras.layers.Dense(1)
    ], name='rmsf_score')

    # Covariance profile subnetwork
    covar_backbone = tf.keras.Sequential([
        tf.keras.layers.Embedding(20, 30, input_length=441),
        tf.keras.layers.Conv1D(72, 10, 1, activation='relu', kernel_initializer=initializer),
        tf.keras.layers.Conv1D(40, 2, 1, activation='relu', kernel_initializer=initializer),
        tf.keras.layers.Conv1D(40, 14, 2, activation='relu', kernel_initializer=initializer),
        tf.keras.layers.GlobalAveragePooling1D(),
    ], name='covar_backbone')
    covar_head = tf.keras.Sequential([
        tf.keras.layers.Dense(4, activation='relu', kernel_initializer=initializer),
        tf.keras.layers.Dense(4, activation='relu', kernel_initializer=initializer),
        tf.keras.layers.Dense(441, activation='linear')
    ], name='covar_head')
    covar_residual = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', kernel_initializer=initializer)  # 16 best
    ], name='covar_residual')
    covar_score = tf.keras.Sequential([
        tf.keras.layers.Dense(1)
    ], name='covar_score')

    # Hydrogen bonding count subnetwork
    hbonds_backbone = copy.copy(rmsf_backbone)
    hbonds_backbone._name = 'hbonds_backbone'
    hbonds_head = copy.copy(covar_head)
    hbonds_head._name = 'hbonds_head'
    hbonds_residual = copy.copy(covar_head)
    hbonds_residual._name = 'hbonds_residual'
    hbonds_score = copy.copy(covar_score)
    hbonds_score._name = 'hbonds_score'

    # Nematic order parameter subnetwork
    nematic_backbone = copy.copy(score_backbone)
    nematic_backbone._name = 'nematic_backbone'
    nematic_head = copy.copy(score_head)
    nematic_head._name = 'nematic_head'
    nematic_score = copy.copy(covar_score)
    nematic_score._name = 'nematic_score'

    # Now we combine them in the desired way depending on settings.obj
    backbones = [score_backbone, covar_backbone, rmsf_backbone]  # , nematic_backbone, hbonds_backbone]
    heads = [score_head, covar_head, rmsf_head]  # , nematic_head, hbonds_head]
    residuals = [None, None, None]  # covar_residual, rmsf_residual]#, None, hbonds_residual]
    scores = [None, covar_score, rmsf_score]  # , nematic_score, hbonds_score]
    objectives = ['score', 'covar', 'rmsf']  # , 'nematic', 'hbonds']

    # Rename each layer by its objective
    for jj in range(len(objectives)):
        kk = 0
        for layer in backbones[jj].layers:
            layer._name = backbones[jj].name + '_' + str(kk)  # renaming for by-name loading later
            kk += 1
        kk = 0
        for layer in heads[jj].layers:
            layer._name = heads[jj].name + '_' + str(kk)
            kk += 1
        if residuals[jj]:
            kk = 0
            for layer in residuals[jj].layers:
                layer._name = heads[jj].name + '_' + str(kk)
                kk += 1

    ### Full residual form ###
    backbone_outputs = [item(input) for item in backbones]
    for jj in range(len(backbone_outputs)):
        if residuals[jj]:
            backbone_outputs[jj] = residuals[jj](backbone_outputs[jj])
    backbones_output = concatenate(backbone_outputs, name='cat_backbone_outputs')

    if settings.obj == 'full_score':
        head_outputs = []
        for jj in range(len(heads)):
            if scores[jj]:
                head_outputs.append(scores[jj](heads[jj](backbones_output)))
            else:
                head_outputs.append(heads[jj](backbones_output))
        heads_output = concatenate(head_outputs, name='cat_head_outputs')
        output = tf.keras.Sequential([
            tf.keras.layers.Dense(1, activation='linear')
        ], name='final_score')(heads_output)
        model = Model(inputs=input, outputs=output)
        for layer in model.layers:
            if 'rmsf' in layer.name or 'covar' in layer.name:
                layer.trainable = False
                if hasattr(layer, 'layers'):
                    for laye in layer.layers:
                        laye.trainable = False
    elif settings.obj == 'score_only':
        model = Model(inputs=input, outputs=score_head(backbones_output))
        for layer in model.layers:
            if 'rmsf' in layer.name or 'covar' in layer.name:
                layer.trainable = False
                if hasattr(layer, 'layers'):
                    for laye in layer.layers:
                        laye.trainable = False
    elif settings.obj == 'covar':
        model = Model(inputs=input, outputs=covar_head(backbones_output))
        for layer in model.layers:
            if not 'covar' in layer.name:
                layer.trainable = False
                if hasattr(layer, 'layers'):
                    for laye in layer.layers:
                        laye.trainable = False
    elif settings.obj == 'rmsf':
        model = Model(inputs=input, outputs=rmsf_head(backbones_output))
        for layer in model.layers:
            if not 'rmsf' in layer.name:
                layer.trainable = False
                if hasattr(layer, 'layers'):
                    for laye in layer.layers:
                        laye.trainable = False
    elif settings.obj == 'covar_score':
        model = Model(inputs=input, outputs=covar_score(covar_head(backbones_output)))
        for layer in model.layers:
            if not layer.name == 'covar_score':
                layer.trainable = False
                if hasattr(layer, 'layers'):
                    for laye in layer.layers:
                        laye.trainable = False
    elif settings.obj == 'rmsf_score':
        model = Model(inputs=input, outputs=rmsf_score(rmsf_head(backbones_output)))
        for layer in model.layers:
            if not layer.name == 'rmsf_score':
                layer.trainable = False
                if hasattr(layer, 'layers'):
                    for laye in layer.layers:
                        laye.trainable = False

    # Reload weights for each component, if available
    print(settings.obj)
    for subnet in backbones + heads + residuals + scores:
        if subnet in model.layers and os.path.exists('model_data/last_' + subnet.name + '.h5'):
            print('loading ' + subnet.name)
            model.get_layer(subnet.name).load_weights('model_data/last_' + subnet.name + '.h5', by_name=True)
        else:
            if subnet:
                print('not loading ' + subnet.name)

    model.summary()
    tf.keras.utils.plot_model(model, to_file='model_data/' + settings.obj + '_graph.png')

    def my_loss_function(y_true, y_pred):
        # Simple custom loss function
        difference = tf.math.subtract(y_true, y_pred)
        scaled_difference = tf.abs(difference ** 2)
        return tf.reduce_mean(scaled_difference, axis=-1)

    def correlationLoss(x, y, axis=-2):
        """Loss function that maximizes the pearson correlation coefficient between the predicted values and the labels,
        while trying to have the same mean and variance"""
        from tensorflow.python.ops import math_ops
        from tensorflow.python.keras import backend as K
        epsilon = hp.Choice('epsilon', values=[0.1, 1.0, 10.0])  # set to small non-zero number to avoid divide-by-zero error
        x = tf.convert_to_tensor(x)
        y = math_ops.cast(y, x.dtype)
        n = tf.cast(tf.shape(x)[axis], x.dtype)
        xsum = tf.reduce_sum(x, axis=axis)
        ysum = tf.reduce_sum(y, axis=axis)
        xmean = xsum / n + epsilon
        ymean = ysum / n + epsilon
        xsqsum = tf.reduce_sum(tf.math.squared_difference(x, xmean), axis=axis)
        ysqsum = tf.reduce_sum(tf.math.squared_difference(y, ymean), axis=axis)
        cov = tf.reduce_sum((x - xmean) * (y - ymean), axis=axis)
        corr = cov / (tf.sqrt(xsqsum * ysqsum))
        sqdif = tf.reduce_sum(tf.math.squared_difference(x, y), axis=axis) / n / tf.sqrt(ysqsum / n)
        return tf.convert_to_tensor(K.mean(tf.constant(1.0, dtype=x.dtype) - corr + (0.01 * sqdif)))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=my_loss_function,
        metrics=[keras.metrics.MeanSquaredError()]
    )

    return model


def create_model(loss, settings):
    # The model is the same every time; what changes is what portion of it is trainable

    initializer = tf.keras.initializers.HeUniform()

    input = Input(shape=(441), name='input')

    # encoding_score: 30
    # filters_1_score: 40
    # width_1_score: 12
    # filters_2_score: 72
    # width_2_score: 6
    # filters_3_score: 72
    # width_3_score: 14
    # dense_1_score: 8
    # dense_2_score: 16

    # encoding: 34
    # conv1_filters: 56
    # conv1_width: 14
    # conv2_filters: 40
    # conv2_width: 23
    # conv3_filters: 32
    # conv3_width: 32

    # encoding: 10
    # conv1_filters: 24
    # conv1_width: 11
    # conv2_filters: 64
    # conv2_width: 32
    # conv3_filters: 64
    # conv3_width: 29

    # encoding: 34
    # conv1_filters: 16
    # conv1_width: 5
    # conv2_filters: 80
    # conv2_width: 17
    # conv3_filters: 80
    # conv3_width: 11

    # Direct score subnetwork
    score_backbone = tf.keras.Sequential([
        tf.keras.layers.Embedding(20, 10, input_length=441),
        tf.keras.layers.Conv1D(24, 11, 2, activation=keras.layers.LeakyReLU(), kernel_initializer=initializer),
        tf.keras.layers.Conv1D(64, 32, 2, activation=keras.layers.LeakyReLU(), kernel_initializer=initializer),
        tf.keras.layers.Conv1D(64, 29, 3, activation=keras.layers.LeakyReLU(), kernel_initializer=initializer),
        tf.keras.layers.GlobalAveragePooling1D(),
        # tf.keras.layers.Dense(16, activation='relu', kernel_initializer=initializer),
    ], name='score_backbone')
    score_head = tf.keras.Sequential([
        tf.keras.layers.Dense(100, activation=keras.layers.LeakyReLU(), kernel_initializer=initializer),
        tf.keras.layers.Dropout(0.2),
        # tf.keras.layers.Dense(16, activation=keras.layers.LeakyReLU(), kernel_initializer=initializer),
        tf.keras.layers.Dense(1, bias_regularizer=tf.keras.regularizers.l2())
    ], name='score_head')

    # RMSF subnetwork
    rmsf_backbone = tf.keras.Sequential([
        tf.keras.layers.Embedding(20, 6, input_length=441),
        tf.keras.layers.Conv1D(8, 8, 2, activation='relu', kernel_initializer=initializer),
        tf.keras.layers.Conv1D(104, 6, 1, activation='relu', kernel_initializer=initializer),
        tf.keras.layers.Conv1D(40, 4, 3, activation='relu', kernel_initializer=initializer),
        tf.keras.layers.GlobalAveragePooling1D(),
        # tf.keras.layers.Dense(16, activation='relu', kernel_initializer=initializer),
    ], name='rmsf_backbone')
    rmsf_head = tf.keras.Sequential([
        tf.keras.layers.Dense(12, activation='relu', kernel_initializer=initializer),
        tf.keras.layers.Dense(32, activation='relu', kernel_initializer=initializer),
        tf.keras.layers.Dense(440, activation='linear', dtype=tf.float32)
    ], name='rmsf_head')
    rmsf_residual = tf.keras.Sequential([
        # tf.keras.layers.Conv1D(8, 8, 2, activation='relu', kernel_initializer=initializer),
        tf.keras.layers.Dense(16, activation='relu', kernel_initializer=initializer)    # 16 best
    ], name='rmsf_residual')
    rmsf_score = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', kernel_initializer=initializer),
        tf.keras.layers.Dense(1)
    ], name='rmsf_score')

    # Covariance profile subnetwork
    covar_backbone = tf.keras.Sequential([
        tf.keras.layers.Embedding(20, 30, input_length=441),
        tf.keras.layers.Conv1D(40, 12, 2, activation=keras.layers.LeakyReLU(), kernel_initializer=initializer),
        tf.keras.layers.Conv1D(72, 6, 2, activation=keras.layers.LeakyReLU(), kernel_initializer=initializer),
        tf.keras.layers.Conv1D(72, 14, 3, activation=keras.layers.LeakyReLU(), kernel_initializer=initializer),
        tf.keras.layers.GlobalAveragePooling1D(),
    ], name='covar_backbone')
    covar_head = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation=keras.layers.LeakyReLU(), kernel_initializer=initializer),
        tf.keras.layers.Dense(16, activation=keras.layers.LeakyReLU(), kernel_initializer=initializer),
        tf.keras.layers.Dense(441, activation='linear', bias_regularizer=tf.keras.regularizers.l2())
    ], name='covar_head')
    covar_residual = tf.keras.Sequential([
        # tf.keras.layers.Conv1D(8, 8, 2, activation='relu', kernel_initializer=initializer),
        tf.keras.layers.Dense(16, activation='relu', kernel_initializer=initializer)    # 16 best
    ], name='covar_residual')
    covar_score = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', kernel_initializer=initializer),
        tf.keras.layers.Dense(1)
    ], name='covar_score')

    # Hydrogen bonding count subnetwork
    hbonds_backbone = copy.copy(rmsf_backbone)
    hbonds_backbone._name = 'hbonds_backbone'
    hbonds_head = copy.copy(covar_head)
    hbonds_head._name = 'hbonds_head'
    hbonds_residual = copy.copy(covar_head)
    hbonds_residual._name = 'hbonds_residual'
    hbonds_score = copy.copy(covar_score)
    hbonds_score._name = 'hbonds_score'

    # Nematic order parameter subnetwork
    nematic_backbone = copy.copy(score_backbone)
    nematic_backbone._name = 'nematic_backbone'
    nematic_head = copy.copy(score_head)
    nematic_head._name = 'nematic_head'
    nematic_score = copy.copy(covar_score)
    nematic_score._name = 'nematic_score'

    # Now we combine them in the desired way depending on settings.obj
    backbones = [score_backbone, covar_backbone, rmsf_backbone]#, nematic_backbone, hbonds_backbone]
    heads = [score_head, covar_head, rmsf_head]#, nematic_head, hbonds_head]
    residuals = [None, None, None]#, None, hbonds_residual]
    scores = [None, covar_score, rmsf_score]#, nematic_score, hbonds_score]
    objectives = ['score', 'covar', 'rmsf']#, 'nematic', 'hbonds']

    # Rename each layer by its objective
    for jj in range(len(objectives)):
        kk = 0
        for layer in backbones[jj].layers:
            layer._name = backbones[jj].name + '_' + str(kk)  # renaming for by-name loading later
            kk += 1
        kk = 0
        for layer in heads[jj].layers:
            layer._name = heads[jj].name + '_' + str(kk)
            kk += 1
        if residuals[jj]:
            kk = 0
            for layer in residuals[jj].layers:
                layer._name = heads[jj].name + '_' + str(kk)
                kk += 1

    ### Partial residual form ###
    # if settings.obj in objectives:
    #     ii = objectives.index(settings.obj)
    # # Set each layer not being trained to untrainable
    # for jj in range(len(objectives)):
    #     if not jj == ii:
    #         for layer in backbones[jj].layers: layer.trainable = False
    # backbone_outputs = [item(input) for item in backbones]
    # for jj in range(len(backbone_outputs)):
    #     if residuals[jj] and not jj == ii:
    #         backbone_outputs[jj] = residuals[jj](backbone_outputs[jj])
    # output = heads[ii](concatenate(backbone_outputs, name='cat_' + str(ii)))

    ### Full residual form ###
    backbone_outputs = [item(input) for item in backbones]
    for jj in range(len(backbone_outputs)):
        if residuals[jj] and not objectives[jj] in settings.obj:
            backbone_outputs[jj] = residuals[jj](backbone_outputs[jj])
    backbones_output = concatenate(backbone_outputs, name='cat_backbone_outputs')

    if settings.obj == 'covar':  # just a test
        model = Model(inputs=input, outputs=covar_head(covar_backbone(input)))
    elif settings.obj == 'covar_score':     # just a test
        head = covar_head
        head.pop()
        model = Model(inputs=input, outputs=covar_score(head(covar_backbone(input))))
        for layer in model.layers:
            if not layer.name == 'covar_score':
                layer.trainable = False
                if hasattr(layer, 'layers'):
                    for laye in layer.layers:
                        laye.trainable = False
    elif settings.obj == 'rmsf':    # just a test
        covar_head_pop = covar_head
        covar_head_pop.pop()
        input_to_head = concatenate([covar_head_pop(covar_backbone(input)), rmsf_backbone(input)])
        model = Model(inputs=input, outputs=rmsf_head(input_to_head))
    elif settings.obj == 'rmsf_score':     # just a test
        head = rmsf_head
        head.pop()
        covar_head_pop = covar_head
        covar_head_pop.pop()
        input_to_head = concatenate([covar_head_pop(covar_backbone(input)), rmsf_backbone(input)])
        model = Model(inputs=input, outputs=rmsf_score(head(input_to_head)))
        for layer in model.layers:
            if not layer.name == 'rmsf_score':
                layer.trainable = False
                if hasattr(layer, 'layers'):
                    for laye in layer.layers:
                        laye.trainable = False
    elif settings.obj == 'full_score':    # just a test
        covar_head_pop = covar_head
        covar_head_pop.pop()
        rmsf_head_pop = rmsf_head
        rmsf_head_pop.pop()
        input_to_head = concatenate([covar_head_pop(covar_backbone(input)), rmsf_head_pop(concatenate([covar_head_pop(covar_backbone(input)), rmsf_backbone(input)])), score_backbone(input)])
        head_output = score_head(input_to_head)
        output = tf.keras.Sequential([
            tf.keras.layers.Dense(1, activation='linear')
        ], name='final_score')(head_output)
        model = Model(inputs=input, outputs=output)
        for layer in model.layers:
            if 'rmsf' in layer.name or 'covar' in layer.name:
                layer.trainable = False
                if hasattr(layer, 'layers'):
                    for laye in layer.layers:
                        laye.trainable = False
    elif settings.obj == 'full_score':
        head_outputs = []
        for jj in range(len(heads)):
            if scores[jj]:
                head_outputs.append(scores[jj](heads[jj](backbones_output)))
            else:
                head_outputs.append(heads[jj](backbones_output))
        heads_output = concatenate(head_outputs, name='cat_head_outputs')
        output = tf.keras.Sequential([
            tf.keras.layers.Dense(1, activation='linear')
        ], name='final_score')(heads_output)
        model = Model(inputs=input, outputs=output)
        for layer in model.layers:
            if 'rmsf' in layer.name or 'covar' in layer.name:
                layer.trainable = False
                if hasattr(layer, 'layers'):
                    for laye in layer.layers:
                        laye.trainable = False
    elif settings.obj == 'score_only':
        model = Model(inputs=input, outputs=score_head(score_backbone(input)))
        for layer in model.layers:
            if 'rmsf' in layer.name or 'covar' in layer.name:
                layer.trainable = False
                if hasattr(layer, 'layers'):
                    for laye in layer.layers:
                        laye.trainable = False
    elif settings.obj == 'covar':
        model = Model(inputs=input, outputs=covar_head(backbones_output))
        for layer in model.layers:
            if not 'covar' in layer.name:
                layer.trainable = False
                if hasattr(layer, 'layers'):
                    for laye in layer.layers:
                        laye.trainable = False
    elif settings.obj == 'rmsf':
        model = Model(inputs=input, outputs=rmsf_head(backbones_output))
        for layer in model.layers:
            if not 'rmsf' in layer.name:
                layer.trainable = False
                if hasattr(layer, 'layers'):
                    for laye in layer.layers:
                        laye.trainable = False
    elif settings.obj == 'covar_score':
        model = Model(inputs=input, outputs=covar_score(covar_head(backbones_output)))
        for layer in model.layers:
            if not layer.name == 'covar_score':
                layer.trainable = False
                if hasattr(layer, 'layers'):
                    for laye in layer.layers:
                        laye.trainable = False
    elif settings.obj == 'rmsf_score':
        model = Model(inputs=input, outputs=rmsf_score(rmsf_head(backbones_output)))
        for layer in model.layers:
            if not layer.name == 'rmsf_score':
                layer.trainable = False
                if hasattr(layer, 'layers'):
                    for laye in layer.layers:
                        laye.trainable = False

    # Reload weights for each component, if available
    print(settings.obj)
    for subnet in backbones + heads + residuals + scores:
        if subnet in model.layers and os.path.exists('model_data/last_' + subnet.name + '.h5'):
            print('loading ' + subnet.name)
            model.get_layer(subnet.name).load_weights('model_data/last_' + subnet.name + '.h5', by_name=True)
        else:
            if subnet:
                print('not loading ' + subnet.name)

    # model = Model(inputs=input, outputs=output)

    # # Load weights from last step
    # if os.path.exists('model_data/last_meta_model.tf'):
    #     model.load_weights('model_data/last_meta_model.tf', by_name=True, skip_mismatch=True)

    # # Constrain each layer to non-negative weights
    # for layer in model.layers:
    #     if hasattr(layer, 'layers'):
    #         for laye in layer.layers:
    #             laye.kernel_constraint = tf.keras.constraints.non_neg
    #     else:
    #         layer.kernel_constraint = tf.keras.constraints.non_neg

    model.summary()
    tf.keras.utils.plot_model(model, to_file='model_data/' + settings.obj + '_graph.png')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=loss,
        metrics=[keras.metrics.MeanSquaredError()],
        run_eagerly=False
    )

    return model, [layer.name for layer in model.layers]


def main(dataset, settings):
    best_r = None
    for training_index in range(max(1, settings.trainings)):
        if not os.path.exists('saved_train_' + settings.obj + '_seed_' + str(settings.seed)):
            train_dataset, test_dataset = get_train_and_test_splits(dataset, settings)
            tf.data.experimental.save(train_dataset, 'saved_train_' + settings.obj + '_seed_' + str(settings.seed))
            tf.data.experimental.save(test_dataset, 'saved_test_' + settings.obj + '_seed_' + str(settings.seed))
        else:
            train_dataset = tf.data.experimental.load('saved_train_' + settings.obj + '_seed_' + str(settings.seed))
            test_dataset = tf.data.experimental.load('saved_test_' + settings.obj + '_seed_' + str(settings.seed))

        def my_loss_function(y_true, y_pred):
            # Simple custom loss function
            difference = tf.math.subtract(y_true, y_pred)
            scaled_difference = tf.abs(difference ** 2)
            return tf.reduce_mean(scaled_difference, axis=-1)

        def correlationLoss(x, y, axis=-2):
            """Loss function that maximizes the pearson correlation coefficient between the predicted values and the labels,
            while trying to have the same mean and variance"""
            from tensorflow.python.ops import math_ops
            from tensorflow.python.keras import backend as K
            epsilon = 10  # set to small non-zero number to avoid divide-by-zero error
            x = tf.convert_to_tensor(x)
            y = math_ops.cast(y, x.dtype)
            n = tf.cast(tf.shape(x)[axis], x.dtype)
            xsum = tf.reduce_sum(x, axis=axis)
            ysum = tf.reduce_sum(y, axis=axis)
            xmean = xsum / n + epsilon
            ymean = ysum / n + epsilon
            xsqsum = tf.reduce_sum(tf.math.squared_difference(x, xmean), axis=axis)
            ysqsum = tf.reduce_sum(tf.math.squared_difference(y, ymean), axis=axis)
            cov = tf.reduce_sum((x - xmean) * (y - ymean), axis=axis)
            corr = cov / (tf.sqrt(xsqsum * ysqsum))
            sqdif = tf.reduce_sum(tf.math.squared_difference(x, y), axis=axis) / n / tf.sqrt(ysqsum / n)
            return tf.convert_to_tensor(K.mean(tf.constant(1.0, dtype=x.dtype) - corr + (0.01 * sqdif)))

        # if 'score' in settings.obj:
        #     loss = correlationLoss
        # else:
        #     loss = my_loss_function
        loss = my_loss_function

        model, subnet_names = create_model(loss, settings)

        SHUFFLE_BUFFER_SIZE = 100
        train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(settings.batch_size)
        test_dataset = test_dataset.batch(settings.batch_size)

        history = run_experiment(model, train_dataset, test_dataset, settings.num_epochs, settings)

        # Get samples to use in validation
        sample = settings.full_size - settings.train_size  # number of samples
        examples, targets = list(test_dataset.unbatch().shuffle(settings.batch_size * 10).batch(sample))[0]
        tr_examples, tr_targets = list(train_dataset.unbatch().shuffle(settings.batch_size * 10).batch(settings.train_size))[0]

        predicted = model(examples).numpy()

        if settings.obj in ['covar', 'rmsf', 'hbonds']:
            rs = []
            for jj in range(len(predicted[0])):
                r = scipy.stats.pearsonr([predicted[ii][jj] for ii in range(sample)], [targets[ii][jj] for ii in range(sample)])
                if np.isnan(r[0]):
                    r = (0, 1)
                rs.append(r)
            this_r = (np.mean([r[0] for r in rs]), np.mean([r[1] for r in rs]))     # this is not rigorous to interpret as an "r" but may still be useful
            # this_r = (history.history['val_loss'][-1], 0)
        else:
            try:
                this_r = scipy.stats.pearsonr([predicted[ii][0] for ii in range(sample)], [targets[ii] for ii in range(sample)])
            except:  # happens for some totally uncorrelated results with zero variance in one dimension
                this_r = (0, 1)

        if (not best_r) or this_r[0] > best_r[0]:
            best_r = this_r
            best_model = model
            best_sample = sample
            best_examples = examples
            best_targets = targets
            best_history = history
            best_tr_examples = tr_examples
            best_tr_targets = tr_targets

            # def freeze(model):
            #     """Freeze model weights in every layer."""
            #     for layer in model.layers:
            #         layer.trainable = True
            #
            #         if isinstance(layer, tf.keras.models.Model):
            #             freeze(layer)
            #
            # freeze(model)
            # model.save_weights('model_data/last_meta_model.tf', save_format='tf')

            for name in subnet_names:
                if not 'cat' in name and not 'input' in name:
                    model.get_layer(name).save_weights('model_data/last_' + name + '.h5')
                    # if model.get_layer(name).trainable:
                    #     print(model.get_layer(name).get_weights())

    if settings.plots and settings.num_epochs > 0:
        plt.semilogy(best_history.history['loss'])
        plt.semilogy(best_history.history['val_loss'])
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    # Normalized scores corresponding to withheld datapoints
    withhold_scores = [0.606158964827018, 0.43302631811274717, 0.36586600331092717, 0.6226576520686639, 0.541914130141165]

    if settings.plots:
        predicted = best_model(best_examples).numpy()
        trained_on = best_model(best_tr_examples).numpy()
    if 'score' in settings.obj and settings.plots:
        colors = []
        for idx in range(best_sample):
            print(f"Predicted: {round(float(predicted[idx][0]), 3)} - Actual: {round(float(targets[idx]), 3)}")
            if round(float(targets[idx]), 7) in [round(val, 7) for val in withhold_scores]:
                colors.append('#ff7f0e')
            else:
                colors.append('#1f77b4')
    print('Pearson r (p): ' + str(best_r))

    if os.path.exists('model_data/saved_rs.pkl'):
        saved_rs = pickle.load(open('model_data/saved_rs.pkl', 'rb'))
    else:
        saved_rs = argparse.Namespace()
    if settings.obj not in saved_rs.__dict__.keys():
        exec('saved_rs.' + settings.obj + ' = [best_r[0]]')
    else:
        exec('saved_rs.' + settings.obj + ' += [best_r[0]]')
    pickle.dump(saved_rs, open('model_data/saved_rs.pkl', 'wb'))

    if settings.plots:
        if settings.obj in ['covar', 'rmsf', 'hbonds']:
            pred = [predicted[idx] for idx in range(best_sample)]
            for jj in range(min(len(pred), 10)):
                plt.scatter(pred[jj], best_targets[jj], s=3)  # , c=colors)
                plt.plot([min(pred[jj]), max(pred[jj])], [min(pred[jj]), max(pred[jj])])
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.show()
                plt.plot(range(len(best_targets[jj])), best_targets[jj])
                plt.plot(range(len(pred[jj])), pred[jj])
                plt.xlabel('Index')
                plt.ylabel('Value')
                plt.show()
        else:
            trnd = [trained_on[idx][0] for idx in range(settings.train_size)]
            pred = [predicted[idx][0] for idx in range(best_sample)]
            plt.scatter(pred, best_targets, s=3, c=colors, zorder=2)
            plt.scatter(trnd, best_tr_targets, s=3, c='#00ff00', zorder=1, alpha=0.3)
            plt.plot([0, 1], [0, 1])
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.show()


if __name__ == "__main__":
    print(tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None))
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    settings = argparse.Namespace()
    mtraj = mdtraj.load('data/one_frame.rst7', top='data/TmAfc_D224G_t200.prmtop')
    settings.seq = [str(atom)[0:3] for atom in mtraj.topology.atoms if (atom.residue.is_protein and (atom.name == 'CA'))]

    # Training settings
    settings.reload_weights = True     # Reload weights from previous run
    settings.trainings = 1              # Number of times to run each training step before taking the best # todo: this is not rigorous without separate validation and test splits
    settings.full_size = 1983 #2363          # Number of total samples available
    settings.train_size = 1683 #2363 - 307  # Number of samples in training set (rest in test set)
    settings.seed = 1111                   # RNG seed to for selecting train and test splits
    settings.batch_size = 16            # Samples per batch

    if not settings.reload_weights:
        if os.path.exists('model_data/last_meta_model.tf'):
            shutil.rmtree('model_data/last_meta_model.tf')
        if os.path.exists('model_data/saved_rs.pkl'):
            os.remove('model_data/saved_rs.pkl')
        # todo: remove subnetwork weight files

    training_schedule = [('full_score', True, 250)]#[('covar', True, 200), ('covar_score', True, 2500), ('rmsf', True, 2500), ('rmsf_score', True, 2500)] * 1 + [('full_score', True, 2500)]
    for training in training_schedule:
        settings.obj = training[0]
        settings.plots = training[1]
        settings.num_epochs = training[2]  # Number of training epochs for each step

        file = 'data/210922_resampled_history.pkl'

        main(pickle.load(open(file, 'rb')), settings)

    saved_rs = pickle.load(open('model_data/saved_rs.pkl', 'rb'))
    if settings.plots:
        for key in saved_rs.__dict__.keys():
            exec('plt.plot(range(len(saved_rs.' + key + ')), saved_rs.' + key + ')')
        plt.xlabel('Training iteration')
        plt.ylabel('Prediction Pearson r')
        plt.legend([str(item) for item in saved_rs.__dict__.keys()])
        plt.show()