import os
import sys
import copy
import pickle
import mdtraj
import argparse
import numpy as np
import scipy.stats
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import *
import tensorflow_datasets as tfds
# import tensorflow_probability as tfp
from isee.main import Thread
from matplotlib import pyplot as plt

def get_train_and_test_splits(dataset, train_size, settings=argparse.Namespace()):
    # # We prefetch with a buffer the same size as the dataset because th dataset
    # is very small and fits into memory.
    # dataset = (
    #     tfds.load(name="wine_quality", as_supervised=True, split="train")
    #     .map(lambda x, y: (x, tf.cast(y, tf.float32)))
    #     .prefetch(buffer_size=dataset_size)
    #     .cache()
    # )
    # # We shuffle with a buffer the same size as the dataset.
    # train_dataset = (
    #     dataset.take(train_size).shuffle(buffer_size=train_size).batch(batch_size)
    # )
    # test_dataset = dataset.skip(train_size).batch(batch_size)

    # Load relevant data from restart.pkl file and parse it into paired (N, k) and (N, ) feature and score matrices
    history = pickle.load(open(dataset, 'rb'))

    # Gather thread history.muts and history.score attributes from each thread together into respective lists
    muts = []
    scores = []
    wt_score = []
    for ii in range(len(history.score)):
        if history.score[ii] and (not history.muts[ii] == ['WT'] and not set(history.muts[ii]) in [set(mut) for mut in muts]) and not max(history.score[ii].covar) == min(history.score[ii].covar):
            # scores.append([history.score[ii].rmsf_lie] + list(history.score[ii].covar) + [item[1] for item in history.score[ii].rmsf])
            if settings.model_obj == 'rmsf':
                scores.append([item[1] for item in history.score[ii].byatom_rmsf][2])
            elif settings.model_obj == 'covar':
                scores.append(list(history.score[ii].covar))
            elif settings.model_obj == 'hbonds':
                scores.append(list(history.score[ii].num_hbonds)[:441])
            else:
                scores.append(history.score[ii].rmsf_lie)
            muts.append(history.muts[ii])     # only get muts with corresponding scores
        elif history.score[ii] and history.muts[ii] == ['WT']:
            if settings.model_obj == 'rmsf':
                wt_score = [item[1] for item in history.score[ii].byatom_rmsf][2]
            elif settings.model_obj == 'covar':
                wt_score = list(history.score[ii].covar)
            elif settings.model_obj == 'hbonds':
                wt_score = list(history.score[ii].num_hbonds)[:441]
            else:
                wt_score = history.score[ii].rmsf_lie

    print('number of distinct mutants: ' + str(len(muts)))

    # # Normalize scores to between 0 and 1
    # print('min: ' + str(min(scores)))
    # print('max: ' + str(max(scores)))
    if settings.model_obj == 'scalar' or settings.model_obj == 'rmsf':
        scores = [(score - min(scores)) / (max(scores) - min(scores)) for score in scores]
    else:
        normscores = []
        for score in scores:
            assert len(score) == len(scores[0])
            assert not max(score) == min(score)
            # normscores.append([(scor - min(score)) / (max(score) - min(score)) for scor in score])
            normscores.append([score[ii] - wt_score[ii] for ii in range(len(score))])
        scores = normscores.copy()

    # Convert each mutants list entry into full sequence-length list of integer-encoded features
    feats = []  # initialize feature vector
    if settings.encoding == 'one_hot' or settings.encoding == 'integer':
        for mut in muts:
            feats.append(integer_encoder(mut, settings.seq))    # get integer encoding
        if settings.encoding == 'one_hot':
            feats = keras.utils.to_categorical(feats)           # convert integer encoding to one-hot encoding
    elif settings.encoding == 'embedded':
        feats = embedded_encoder(muts, settings.seq)  # learn an embedding
    elif settings.encoding == 'categorical':
        for mut in muts:
            feats.append(categorical_encoder(mut, settings.seq))
    else:
        raise RuntimeError('unrecognized encoding: ' + settings.encoding)

    # Subdivide into training and testing datasets based on train_size argument and return
    rng = np.random.default_rng(settings.random_seed)
    indices = [ind for ind in range(len(feats))]
    rng.shuffle(indices)
    test_split = (len(feats) - train_size) / len(feats)

    # A kludge for now that allows me to specify mutants to withold for the test set
    pre_indices = copy.copy(indices)
    to_withold = [['64ASP', '386SER', '394ALA'],['64ASP', '386SER'],['64ASP', '394ALA', '407PRO', '423PRO'],['64ASP', '386SER', '360VAL', '389ASN'], ['64ASP', '386SER', '422THR']]
    indices_of_witholding = [muts.index(mut) for mut in to_withold] # order of single muts within each mut matters for this implementation
    print(indices_of_witholding)
    for index in indices_of_witholding:
        print(muts[index])
    to_exchange = []
    for ii in indices_of_witholding:
        if indices.index(ii) < int(len(feats) * (1 - test_split)):  # if ii is in the train split
            to_exchange.append(ii)
    for jj in range(len(to_exchange)):
        temp = indices[-1 * (jj + 1)]
        original_index = indices.index(to_exchange[jj])
        indices[-1 * (jj + 1)] = to_exchange[jj]
        indices[original_index] = temp
        print('moving mutant at index ' + str(to_exchange[jj]) + ' (' + str(muts[to_exchange[jj]]) + ') to index: ' + str(-1 * (jj + 1)))

    muts = [muts[ii] for ii in indices]
    feats = [feats[ind] for ind in indices]
    scores = [scores[ind] for ind in indices]

    assert (all([item in indices for item in pre_indices]))
    assert (all([mut in muts[int(len(feats) * (1 - test_split)):] for mut in to_withold]))

    feats_train = np.array(feats[:int(len(feats) * (1 - test_split))])
    scores_train = np.array(scores[:int(len(feats) * (1 - test_split))])
    feats_test = np.array(feats[int(len(feats) * (1 - test_split)):])
    scores_test = np.array(scores[int(len(feats) * (1 - test_split)):])
    # assert all([keras.utils.to_categorical(integer_encoder(mut, settings.seq)) in feats_test for mut in to_withold])

    train_dataset = tf.data.Dataset.from_tensor_slices((feats_train, scores_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((feats_test, scores_test))

    return train_dataset, test_dataset

def integer_encoder(muts, wt_seq):
    # Encode sequence with mutations as a list of integers

    all_resnames = ['ARG', 'HIS', 'LYS', 'ASP', 'GLU', 'SER', 'THR', 'ASN', 'GLN', 'CYS', 'GLY', 'PRO', 'ALA',
                    'VAL', 'ILE', 'LEU', 'MET', 'PHE', 'TYR', 'TRP']    # all resnames in enconding order

    if muts == ['WT']:   # special case
        muts = []        # no mutation applied

    values = copy.copy(wt_seq)
    for mut in muts:
        values[int(mut[:-3]) - 1] = mut[-3:]

    int_encoded = []
    values_index = -1
    for item in values:

        int_encoded += [all_resnames.index(item)]

        # values_index += 1
        # if wt_seq[values_index] == item:
        #     int_encoded += [0]
        # else:
        #     int_encoded += [all_resnames.index(item) + 1]

    return int_encoded

def categorical_encoder(muts, wt_seq):
    # Encode sequence with mutations as a list of resnames

    all_resnames = ['ARG', 'HIS', 'LYS', 'ASP', 'GLU', 'SER', 'THR', 'ASN', 'GLN', 'CYS', 'GLY', 'PRO', 'ALA',
                    'VAL', 'ILE', 'LEU', 'MET', 'PHE', 'TYR', 'TRP']    # all resnames in encoding order

    if muts == ['WT']:   # special case
        muts = []        # no mutation applied

    values = copy.copy(wt_seq)
    for mut in muts:
        values[int(mut[:-3]) - 1] = mut[-3:]

    return values

def embedded_encoder(muts, wt_seq):
    # Learn an embedding for the given mutants in terms of the following features:
    #   Residue index
    #   Geometric positions (x, y, z coordinates) of alpha carbons  # todo: implement
    #   Sidechain charge at pH 7
    #   Molecular weight
    #   Sidechain hydrophobicity
    #   Sidechain aromaticity

    embedding_size = 1

    # First, define by-residue-name characteristics
    resnames = ['ARG', 'HIS', 'LYS', 'ASP', 'GLU', 'SER', 'THR', 'ASN', 'GLN', 'CYS', 'GLY', 'PRO', 'ALA',
                'VAL', 'ILE', 'LEU', 'MET', 'PHE', 'TYR', 'TRP']    # all resnames in enconding order
    charges =  [1, 1, 1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]   # charges in same order
    mws =      [174, 155, 146, 133, 146, 105, 119, 132, 146, 121, 75, 115, 89, 117, 131, 131, 149, 165, 181, 204]
    hydrophobicity = [0, 0.165, 0.283, 0.028, 0.043, 0.359, 0.450, 0.236, 0.251, 0.680, 0.501, 0.711, 0.616,
                      0.825, 0.943, 0.943, 0.738, 1, 0.880, 0.878]
    aromaticity = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]

    def vocabularize(raw):
        # Convert a "raw" list of values into a categorical vocabulary for use in Embedding
        vocab_index = 0
        vocab = []
        for ii in range(len(raw)):
            if raw[ii] in raw[:ii]:
                vocab.append(vocab[raw.index(raw[ii])])
            else:
                vocab.append(vocab_index)
                vocab_index += 1
        return vocab

    # Build an Input for each feature
    charge_input = keras.Input(name='charge', shape=[441])
    mw_input = keras.Input(name='mw', shape=[441])
    hydrophobicity_input = keras.Input(name='hydrophobicity', shape=[1])
    aromaticity_input = keras.Input(name='aromaticity', shape=[1])
    # coords_input = keras.Input(name='coords', shape=[3])

    # Build an Embedding layer for each Input
    charge_embedding = keras.layers.Embedding(name='charge_embedding',
                               input_dim=max(vocabularize(charges)) + 1,
                               input_length=len(wt_seq),
                               output_dim=embedding_size)(charge_input)
    mw_embedding = keras.layers.Embedding(name='mw_embedding',
                               input_dim=max(vocabularize(mws)) + 1,
                               input_length=len(wt_seq),
                               output_dim=embedding_size)(mw_input)
    # hydrophobicity_embedding = keras.layers.Embedding(name='hydrophobicity_embedding',
    #                            input_length=len(wt_seq),
    #                            output_dim=embedding_size)(hydrophobicity_input)
    # aromaticity_embedding = keras.layers.Embedding(name='aromaticity_embedding',
    #                            input_length=len(wt_seq),
    #                            output_dim=embedding_size)(aromaticity_input)
    # coords_embedding = keras.layers.Embedding(name='coords_embedding',
    #                            input_dim=len(resnames),
    #                            output_dim=embedding_size)(coords_input)

    # Merge the layers with a dot product along the second axis
    merged = keras.layers.Dot(name='dot_product', normalize=True, axes=2)([charge_embedding, mw_embedding])#, hydrophobicity_embedding, aromaticity_embedding])

    # Reshape to be a single number (shape will be (None, 1))
    # merged = keras.layers.Reshape(target_shape=[1])(merged)

    # Output neuron
    out = keras.layers.Dense(1, activation='sigmoid')(merged)
    model = keras.Model(inputs=[charge_input, mw_input], outputs=out)
    # model = keras.Model(inputs=[charge_input, mw_input, hydrophobicity_input, aromaticity_input], outputs=out)

    # Minimize binary cross entropy
    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

    print(model.summary())

    charges = vocabularize(charges)
    mws = vocabularize(mws)

    # Featurize each item in muts
    featurized_muts = []
    for mut in muts:
        # Build full mutated sequence for this mut
        values = copy.copy(wt_seq)
        for mu in mut:
            values[int(mu[:-3]) - 1] = mu[-3:]

        featurized_muts.append([[charges[resnames.index(res)] for res in values],
                                [mws[resnames.index(res)] for res in values]])

    featurized_muts = np.array(featurized_muts)
    featurized_muts = {'charge': featurized_muts[:, 0], 'mw': featurized_muts[:, 1]}

    embeddings = model.predict(featurized_muts)

    return embeddings


"""
## Compile, train, and evaluate the model
"""

def run_experiment(model, loss, train_dataset, test_dataset, num_epochs):

    model.compile(
        optimizer='adam',
        loss=loss,
        metrics=[keras.metrics.MeanSquaredError()],
    )

    print("Start training the model...")
    history = model.fit(train_dataset, epochs=num_epochs, validation_data=test_dataset)
    print("Model training finished.")
    _, rmse = model.evaluate(train_dataset, verbose=0)
    print(f"Train RMSE: {round(rmse, 3)}")

    print("Evaluating model performance...")
    _, rmse = model.evaluate(test_dataset, verbose=0)
    print(f"Test RMSE: {round(rmse, 3)}")

    return history

def create_model(batch_size, obj='scalar', vec_to_scl=False):
    input_shape = (batch_size, 441, 20)

    if obj == 'scalar':
        if vec_to_scl:
            rmsf_model = tf.keras.Sequential([
                tf.keras.layers.Input(batch_input_shape=input_shape),
                tf.keras.layers.Conv1D(256, 10, 2, activation='relu', trainable=False),  # relu performs much better than linear
                # tf.keras.layers.MaxPool1D(2),
                tf.keras.layers.BatchNormalization(1, trainable=False),  # subjectively performing much better than dropout
                tf.keras.layers.Conv1D(128, 8, 2, activation='relu', trainable=False),
                # tf.keras.layers.MaxPool1D(2),
                tf.keras.layers.BatchNormalization(1, trainable=False),
                tf.keras.layers.Conv1D(128, 6, 3, activation='relu', trainable=False),
                # tf.keras.layers.MaxPool1D(2),
                tf.keras.layers.GlobalAveragePooling1D(trainable=False),  # global avg. pooling or flatten --> dense "head" network
                tf.keras.layers.Dense(512, trainable=False),
                tf.keras.layers.Dense(512, trainable=False),
                tf.keras.layers.Dense(440, trainable=False)
            ])
            rmsf_model.load_weights('last_rmsf_model.keras')

            # Define single input
            input = Input(batch_input_shape=input_shape)

            x = rmsf_model(input)
            
            covar_model = tf.keras.Sequential([
                tf.keras.layers.Input(batch_input_shape=input_shape),
                tf.keras.layers.Conv1D(256, 10, 2, activation='relu', trainable=False),  # relu performs much better than linear
                # tf.keras.layers.MaxPool1D(2),
                tf.keras.layers.BatchNormalization(1, trainable=False),  # subjectively performing much better than dropout
                tf.keras.layers.Conv1D(128, 8, 2, activation='relu', trainable=False),
                # tf.keras.layers.MaxPool1D(2),
                tf.keras.layers.BatchNormalization(1, trainable=False),
                tf.keras.layers.Conv1D(128, 6, 3, activation='relu', trainable=False),
                # tf.keras.layers.MaxPool1D(2),
                tf.keras.layers.GlobalAveragePooling1D(trainable=False),  # global avg. pooling or flatten --> dense "head" network
                tf.keras.layers.Dense(512, trainable=False),
                tf.keras.layers.Dense(512, trainable=False),
                tf.keras.layers.Dense(441, trainable=False)
            ])
            covar_model.load_weights('last_covar_model.keras')

            y = covar_model(input)

            # the second branch
            scalar_model = tf.keras.Sequential([
                tf.keras.layers.Input(batch_input_shape=input_shape),
                tf.keras.layers.Conv1D(256, 10, 2, activation='relu', trainable=False),  # relu performs much better than linear
                tf.keras.layers.MaxPool1D(2, trainable=False),
                tf.keras.layers.BatchNormalization(1, trainable=False),      # subjectively performing much better than dropout
                tf.keras.layers.Conv1D(128, 8, 2, activation='relu', trainable=False),
                tf.keras.layers.MaxPool1D(2, trainable=False),
                tf.keras.layers.BatchNormalization(1, trainable=False),
                tf.keras.layers.Conv1D(128, 6, 3, activation='relu', trainable=False),
                # tf.keras.layers.MaxPool1D(2),
                tf.keras.layers.GlobalAveragePooling1D(trainable=False),   # global avg. pooling or flatten --> dense "head" network
                tf.keras.layers.Dense(128, trainable=False),
                tf.keras.layers.Dense(64, trainable=False),
                tf.keras.layers.Dense(1, trainable=False)
            ])
            scalar_model.load_weights('last_scalar_model.keras')

            z = scalar_model(input)

            # combine the output of the two branches
            combined = concatenate([x, y, z])

            # apply a FC layer and then a regression prediction on the combined outputs
            c = Dense(256, activation="relu")(combined)
            c = Dense(64, activation="linear")(c)
            c = Dense(1, activation="linear")(c)

            # model will accept the inputs of the two branches and then output a single value
            model = Model(inputs=input, outputs=c)

        else:
            model = tf.keras.Sequential([
                tf.keras.layers.Embedding(20, 30, input_length=441),
                # tf.keras.layers.Input(batch_input_shape=input_shape),
                tf.keras.layers.Conv1D(256, 10, 2, activation='relu'),  # relu performs much better than linear
                tf.keras.layers.MaxPool1D(2),
                tf.keras.layers.BatchNormalization(1),      # subjectively performing much better than dropout
                tf.keras.layers.Conv1D(128, 8, 2, activation='relu'),
                tf.keras.layers.MaxPool1D(2),
                tf.keras.layers.BatchNormalization(1),
                tf.keras.layers.Conv1D(128, 6, 3, activation='relu'),
                # tf.keras.layers.MaxPool1D(2),
                tf.keras.layers.GlobalAveragePooling1D(),   # global avg. pooling or flatten --> dense "head" network
                tf.keras.layers.Dense(128),
                tf.keras.layers.Dense(64),
                tf.keras.layers.Dense(1)
            ])

    elif obj == 'rmsf':
        model = tf.keras.Sequential([
            tf.keras.layers.Input(batch_input_shape=input_shape),
            tf.keras.layers.Conv1D(256, 10, 2, activation='relu'),  # relu performs much better than linear
            tf.keras.layers.MaxPool1D(2),
            tf.keras.layers.BatchNormalization(1),      # subjectively performing much better than dropout
            tf.keras.layers.Conv1D(128, 8, 2, activation='relu'),
            tf.keras.layers.MaxPool1D(2),
            tf.keras.layers.BatchNormalization(1),
            tf.keras.layers.Conv1D(128, 6, 3, activation='relu'),
            # tf.keras.layers.MaxPool1D(2),
            tf.keras.layers.GlobalAveragePooling1D(),   # global avg. pooling or flatten --> dense "head" network
            # tf.keras.layers.Dense(512),
            # tf.keras.layers.Dense(512),
            # tf.keras.layers.Dense(440)
            tf.keras.layers.Dense(128),
            tf.keras.layers.Dense(64),
            tf.keras.layers.Dense(1)
        ])
        d_n1,d_n2,d_n3 = [model.layers[ii].get_weights() for ii in [-1, -2, -3]]
        model.load_weights('last_scalar_model.keras')
        # model.layers[-3].set_weights(d_n3)
        # model.layers[-2].set_weights(d_n2)
        # model.layers[-1].set_weights(d_n1)
    elif obj in ['covar', 'hbonds']:
        model = tf.keras.Sequential([
            tf.keras.layers.Input(batch_input_shape=input_shape),
            tf.keras.layers.Conv1D(2048, 10, 2, activation='relu'),  # relu performs much better than linear
            # tf.keras.layers.MaxPool1D(2),
            tf.keras.layers.BatchNormalization(1),      # subjectively performing much better than dropout
            tf.keras.layers.Conv1D(1024, 10, 2, activation='relu'),  # relu performs much better than linear
            # tf.keras.layers.MaxPool1D(2),
            tf.keras.layers.BatchNormalization(1),  # subjectively performing much better than dropout
            tf.keras.layers.Conv1D(512, 8, 2, activation='relu'),
            # tf.keras.layers.MaxPool1D(2),
            tf.keras.layers.BatchNormalization(1),
            tf.keras.layers.Conv1D(512, 6, 3, activation='relu'),
            # # tf.keras.layers.MaxPool1D(2),
            tf.keras.layers.GlobalAveragePooling1D(),   # global avg. pooling or flatten --> dense "head" network
            tf.keras.layers.Dense(2048),
            tf.keras.layers.Dense(2048),
            tf.keras.layers.Dense(2048),
            tf.keras.layers.Dense(1024),
            tf.keras.layers.Dense(1024),
            tf.keras.layers.Dense(1024),
            tf.keras.layers.Dense(512),
            tf.keras.layers.Dense(512),
            tf.keras.layers.Dense(512),
            tf.keras.layers.Dense(441)
        ])
    else:
        raise RuntimeError('unknown model objective type: ' + str(obj))

    model.summary()

    return model

def main(dataset, settings):
    best_r = None
    for training_index in range(max(1, settings.repeat_trainings)):
        full_size = 1512
        train_size = 1460
        train_dataset, test_dataset = get_train_and_test_splits(dataset, train_size, settings)

        num_epochs = 300
        BATCH_SIZE = 10

        # Implement random_forest support
        if settings.model_type == 'random_forest':
            import tensorflow_decision_forests as tfdf
            from wurlitzer import sys_pipes
            import pandas as pd
            model = tfdf.keras.RandomForestModel(task=tfdf.keras.Task.REGRESSION, num_trees=300)
            model.compile(metrics=["mse"])

            # Convert datasets to pandas dataframe
            ii = 0
            with open('train.csv', 'w') as f:
                f.write('score,' + ','.join([str(ii) for ii in range(441)]))
                f.write('\n')
                for item in train_dataset:
                    f.write(str(tf.keras.backend.get_value(item[1])) + ',' + ','.join([item.decode() for item in tf.keras.backend.get_value(item[0])]))
                    f.write('\n')
                    ii += 1
            train_dataset = pd.read_csv('train.csv', header=0)
            train_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(train_dataset, label='score', task=tfdf.keras.Task.REGRESSION)
            with open('test.csv', 'w') as f:
                f.write('score,' + ','.join([str(ii) for ii in range(441)]))
                f.write('\n')
                for item in test_dataset:
                    f.write(str(tf.keras.backend.get_value(item[1])) + ',' + ','.join([item.decode() for item in tf.keras.backend.get_value(item[0])]))
                    f.write('\n')
                    ii += 1
            test_dataset = pd.read_csv('test.csv', header=0)
            test_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(test_dataset, label='score', task=tfdf.keras.Task.REGRESSION)

            model.fit(train_dataset)
            # os.remove('temp.csv')
            # with sys_pipes():
            #     model.fit(x=train_dataset)
            model.summary()
            evaluation = model.evaluate(test_dataset, return_dict=True)
            print(evaluation)
            print(f"MSE: {evaluation['mse']}")

            # Get samples to use in validation
            sample = full_size - train_size  # number of samples
            examples, targets = list(test_dataset.unbatch().shuffle(BATCH_SIZE * 10).batch(sample))[0]

            predicted = model(examples).numpy()
            r = scipy.stats.pearsonr([predicted[ii][0] for ii in range(sample)], [targets[ii] for ii in range(sample)])
            predicted = model(examples).numpy()
            for idx in range(sample):
                print(f"Predicted: {round(float(predicted[idx][0]), 3)} - Actual: {round(float(targets[idx]), 3)}")
            print('Pearson r (p): ' + str(r))

            sys.exit()

        # If not random_forest, proceed with CNN
        model = create_model(BATCH_SIZE, obj=settings.model_obj, vec_to_scl=settings.vec_to_scl)

        SHUFFLE_BUFFER_SIZE = 20    # 100
        train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
        test_dataset = test_dataset.batch(BATCH_SIZE)

        # Reload previous model if desired
        if os.path.exists('last_' + settings.model_obj + '_model.keras') and settings.reload_weights:
            model.load_weights('last_' + settings.model_obj + '_model.keras')
        # model.load_weights('/Users/tburgin/miniconda3/lib/python3.7/site-packages/gym/envs/tburgin_custom/tmafc_model.keras')

        history = run_experiment(model, keras.losses.MeanSquaredError(), train_dataset, test_dataset, num_epochs)

        # Get samples to use in validation
        sample = full_size - train_size    # number of samples
        examples, targets = list(test_dataset.unbatch().shuffle(BATCH_SIZE * 10).batch(sample))[0]

        if settings.model_obj in ['rmsf', 'covar', 'hbonds'] or not settings.vec_to_scl:
            model.save('last_' + settings.model_obj + '_model.keras')
        else:
            model.save('last_' + settings.model_obj + '_vec2scl_model.keras')

        predicted = model(examples).numpy()

        if settings.model_obj in ['covar']:
            rs = []
            for jj in range(len(predicted[0])):
                rs.append(scipy.stats.pearsonr([predicted[ii][jj] for ii in range(sample)], [targets[ii][jj] for ii in range(sample)]))
            this_r = (np.mean([r[0] for r in rs]), np.mean([r[1] for r in rs]))
        else:
            this_r = scipy.stats.pearsonr([predicted[ii][0] for ii in range(sample)], [targets[ii] for ii in range(sample)])

        if best_r == None or this_r > best_r:
            best_r = this_r
            best_model = model
            best_sample = sample
            best_examples = examples
            best_targets = targets
            best_history = history

    # best_model.save('last_model.keras')  # dump model for recovery later if desired

    # print(best_history.history)
    if settings.plots:
        plt.plot(best_history.history['loss'])
        plt.plot(best_history.history['val_loss'])
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    predicted = best_model(best_examples).numpy()
    if (settings.model_obj == 'scalar' or settings.model_obj == 'rmsf') and settings.plots:
        colors = []
        for idx in range(best_sample):
            print(f"Predicted: {round(float(predicted[idx][0]), 3)} - Actual: {round(float(targets[idx]), 3)}")
            if round(float(targets[idx]), 7) in [round(val, 7) for val in [0.246499653, 0.277258577, 0.292320875, 0.246250577, 0.200327526]]:
                colors.append('#ff7f0e')
            else:
                colors.append('#1f77b4')
    print('Pearson r (p): ' + str(best_r))

    if settings.plots:
        if settings.model_obj in ['covar', 'hbonds']:
            pred = [predicted[idx] for idx in range(best_sample)]
            for jj in range(len(pred)):
                plt.scatter(pred[jj], best_targets[jj], s=3)#, c=colors)
                plt.plot([min(pred[jj]), max(pred[jj])], [min(pred[jj]), max(pred[jj])])
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.show()
                plt.plot(range(len(pred[jj])),pred[jj])
                plt.plot(range(len(best_targets[jj])), best_targets[jj])
                plt.xlabel('Index')
                plt.ylabel('Covariance')
                plt.show()
        else:
            pred = [predicted[idx][0] for idx in range(best_sample)]
            plt.scatter(pred, best_targets, s=3, c=colors)
            plt.plot([0, 1], [0, 1])
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.show()

if __name__ == "__main__":
    settings = argparse.Namespace()
    mtraj = mdtraj.load('data/one_frame.rst7', top='data/TmAfc_D224G_t200.prmtop')
    settings.seq = [str(atom)[0:3] for atom in mtraj.topology.atoms if (atom.residue.is_protein and (atom.name == 'CA'))]
    print(settings.seq)

    settings.reload_weights = False
    settings.encoding = 'integer'
    settings.repeat_trainings = 1
    settings.model_type = 'cnn'

    # Set an rng seed to for selecting train and test splits
    settings.random_seed = 2    # set to None for a new pseudorandom seed each time

    training_schedule = [('scalar', False, True)] #[('rmsf', False, False), ('covar', False, False), ('scalar', False, False), ('scalar', True, True)]
    for training in training_schedule:
        settings.model_obj = training[0]
        settings.vec_to_scl = training[1]
        settings.plots = training[2]

        main('data/resampled_history.pkl', settings)
