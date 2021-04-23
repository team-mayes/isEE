import sys
import copy
import pickle
import mdtraj
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
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
    allthreads = pickle.load(open(dataset, 'rb'))

    # Gather thread history.muts and history.score attributes from each thread together into respective lists
    muts = []
    scores = []
    for thread in allthreads:
        scores += thread.history.score
        muts += thread.history.muts[:len(thread.history.score)]     # only get muts with corresponding scores

    # # Normalize scores to between 0 and 1
    scores = [(score - min(scores)) / (max(scores) - min(scores)) for score in scores]

    # Convert each mutants list entry into full sequence-length list of integer-encoded features
    feats = []
    for mut in muts:
        feats.append(integer_encoder(mut, settings.seq))

    # Convert integer encoding to one-hot encoding
    feats = keras.utils.to_categorical(feats)

    # Subdivide into training and testing datasets based on train_size argument and return
    rng = np.random.default_rng()
    indices = [ind for ind in range(len(feats))]
    rng.shuffle(indices)
    feats = [feats[ind] for ind in indices]
    scores = [scores[ind] for ind in indices]
    test_split = (len(feats) - train_size) / len(feats)

    feats_train = np.array(feats[:int(len(feats) * (1 - test_split))])
    scores_train = np.array(scores[:int(len(feats) * (1 - test_split))])
    feats_test = np.array(feats[int(len(feats) * (1 - test_split)):])
    scores_test = np.array(scores[int(len(feats) * (1 - test_split)):])

    train_dataset = tf.data.Dataset.from_tensor_slices((feats_train, scores_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((feats_test, scores_test))

    return train_dataset, test_dataset

def integer_encoder(muts, wt_seq):
    # Encode sequence with mutations as a list of integers from 1 to 20

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

        values_index += 1
        if wt_seq[values_index] == item:
            int_encoded += [0]
        else:
            ind = 0
            for resname in all_resnames:
                ind += 1
                if item == resname:
                    int_encoded += [ind] #[1]
                    break

    return int_encoded

"""
## Compile, train, and evaluate the model
"""

def run_experiment(model, loss, train_dataset, test_dataset, num_epochs):

    model.compile(
        optimizer='adam',
        loss=loss,
        metrics=[keras.metrics.RootMeanSquaredError()],
    )

    print("Start training the model...")
    model.fit(train_dataset, epochs=num_epochs, validation_data=test_dataset)
    print("Model training finished.")
    _, rmse = model.evaluate(train_dataset, verbose=0)
    print(f"Train RMSE: {round(rmse, 3)}")

    print("Evaluating model performance...")
    _, rmse = model.evaluate(test_dataset, verbose=0)
    print(f"Test RMSE: {round(rmse, 3)}")

def create_model(): #7, 9, 21, 49, 63, 147
    input_shape = (40, 441, 21)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(batch_input_shape=input_shape),
        tf.keras.layers.Conv1D(256, 10, 2, activation='relu'),
        tf.keras.layers.MaxPool1D(2),
        tf.keras.layers.Conv1D(128, 8, 2, activation='relu'),
        tf.keras.layers.MaxPool1D(2),
        tf.keras.layers.Conv1D(128, 6, 3, activation='relu'),
        tf.keras.layers.MaxPool1D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128),
        tf.keras.layers.Dense(64)
    ])

    model.summary()

    return model

def main(dataset, settings):
    train_size = 2000
    train_dataset, test_dataset = get_train_and_test_splits(dataset, train_size, settings)

    num_epochs = 100
    model = create_model()

    BATCH_SIZE = 40
    SHUFFLE_BUFFER_SIZE = 100
    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    # Reload previous model if desired
    model.load_weights('last_model.keras')

    run_experiment(model, keras.losses.MeanSquaredError(), train_dataset, test_dataset, num_epochs)

    model.save('last_model.keras')    # dump model for recovery later if desired

    # Get samples to use in validation
    sample = 150    # number of samples
    examples, targets = list(train_dataset.unbatch().shuffle(BATCH_SIZE * 10).batch(sample))[0]

    predicted = model(examples).numpy()
    for idx in range(sample):
        print(f"Predicted: {round(float(predicted[idx][0]), 3)} - Actual: {round(float(targets[idx]), 3)}")

    pred = [predicted[idx][0] for idx in range(sample)]
    # pred = [(item - min(pred)) / (max(pred) - min(pred)) for item in pred]

    plt.scatter(pred, targets, s=1)
    plt.plot([0, 1], [0, 1])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

if __name__ == "__main__":
    settings = argparse.Namespace()
    mtraj = mdtraj.load('data/one_frame.rst7', top='data/TmAfc_D224G_t200.prmtop')
    settings.seq = [str(atom)[0:3] for atom in mtraj.topology.atoms if (atom.residue.is_protein and (atom.name == 'CA'))]
    print(settings.seq)
    main('data/spoof_low_noise_restart.pkl', settings)
