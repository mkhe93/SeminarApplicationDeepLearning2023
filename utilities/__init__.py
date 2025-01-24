import pickle
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import tensorflow as tf
from tqdm import tqdm
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# ----------------------------------------------------------------------------------------------------
# Load dataset
# ----------------------------------------------------------------------------------------------------
def load_dataset(path: str, max_items: int = None) -> tuple:
    """
        Loads a dataset from a pickle file.

        Args:
            path (str): The path to the pickle file.
            max_items (int, optional): The maximum number of items to load from the dataset. Defaults to None.

        Returns:
            tuple: A tuple containing the loaded data, labels, fids, velocities, and angles.

        Created by ChatGPT.
    """
    with open(path, 'rb') as file:
        data, labels, fids, velocities, angles = pickle.load(file)

    if max_items:
        return data[:max_items], labels[:max_items], np.array(fids[:max_items]), np.array(velocities[:max_items]), np.array(angles[:max_items])

    return data, labels, np.array(fids), np.array(velocities), np.array(angles)


# ----------------------------------------------------------------------------------------------------
# Data transformations
# ----------------------------------------------------------------------------------------------------
def pad_data(data: list, size: int) -> np.ndarray:
    """
        Pads the given data list to a specified size along the first dimension.

        Args:
            data (list): The input data list to be padded.
            size (int): The desired size of the first dimension after padding.

        Returns:
            numpy.ndarray: The padded data as a NumPy array.

        Created by ChatGPT.
    """
    data_size = len(data)
    padded_data = np.zeros((data_size, size, 6), dtype=np.float16)

    for idx in range(data_size):
        sample_length = len(data[idx])
        start_idx = (size - sample_length) // 2

        padded_data[idx, start_idx:start_idx + sample_length, :] = data[idx]

    return padded_data


# --------------------------------------------------
# Normalization
# --------------------------------------------------
def normalize_sample(sample: np.ndarray):
    """
        Normalizes a sample by subtracting the mean and dividing by the maximum absolute value.

        Args:
            sample (numpy.ndarray): Input sample to be normalized.

        Returns:
            None

        Created by ChatGPT.
    """
    sample -= sample.mean(axis=0)
    sample /= np.max([np.abs(sample.min(axis=0)), np.abs(sample.max(axis=0))], axis=0)


def normalize_data(data: list):
    """
        Normalizes the given data array by subtracting the mean and dividing by the maximum absolute value.

        Args:
            data (list): The input data array to be normalized.

        Returns:
            None

        Created by ChatGPT.
    """
    for idx in range(len(data)):
        normalize_sample(data[idx])


# --------------------------------------------------
# Compression / (Interpolation)
# --------------------------------------------------
def compress_sample(sample: np.ndarray, size: int) -> np.ndarray:
    """
        Compresses the given sample array to a specified size using interpolation.

        Args:
            sample (numpy.ndarray): The input sample array.
            size (int): The desired size of the compressed sample.

        Returns:
            numpy.ndarray: The compressed sample array.

        Created by ChatGPT.
    """
    sample_length = sample.shape[0]
    interpolation_function = scipy.interpolate.interp1d(np.arange(sample_length), sample, axis=0)
    return interpolation_function(np.linspace(0, sample_length - 1, size))


def compress_data(data: list, size: int, use_tqdm: bool = False):
    """
        Compresses the data samples in the given list to a specified size using interpolation.

        Args:
            data (list): The list of data samples.
            size (int): The desired size of the compressed samples.
            use_tqdm (bool, optional): Whether to use tqdm to display a progress bar (default: False).

        Returns:
            None

        Created by ChatGPT.
    """
    iterator = range(len(data))
    if use_tqdm:
        iterator = tqdm(iterator)

    for idx in iterator:
        data[idx] = compress_sample(data[idx], size)


# ----------------------------------------------------------------------------------------------------
# Plotting functions
# ----------------------------------------------------------------------------------------------------
def plot_sample(sample: np.ndarray):
    """
        Plots a sample of signals related to speed bump crossing.

        Args:
            sample (numpy.ndarray): The sample data containing signals.

        Returns:
            None

        Created by ChatGPT.
    """
    labels = [
        r"$amplitude_a$",
        r"$amplitude_b$",
        r"$amplitude_c$",
        r"$amplitude_d$",
        r"$amplitude_e$",
        r"$amplitude_f$"
    ]

    fig, axs = plt.subplots(3, 2, figsize=(25, 8))
    sample_points = np.arange(sample.shape[0])

    for signal in range(6):
        x, y = signal % 3, signal // 3

        axs[x, y].plot(sample_points, sample[:, signal], label='Sample')
        axs[x, y].set(ylabel=labels[signal])
        axs[x, y].legend()

    axs[2, 0].set(xlabel="Time samples (100HZ sampling rate)")
    axs[2, 1].set(xlabel="Time samples (100HZ sampling rate)")
    fig.suptitle("Measurement of speed bump crossing")


def plot_batch(batch: list):
    """
        Plots a batch of samples.

        Args:
            batch (list): List of samples.

        Returns:
            None

        Created by ChatGPT.
    """
    batch_size = len(batch)
    num_rows = batch_size // 2 + batch_size % 2

    fig, axs = plt.subplots(num_rows, 2, figsize=(20, batch_size))

    for idx, sample in enumerate(batch):
        row = idx // 2
        col = idx % 2
        axs[row, col].plot(sample)

    if batch_size % 2 != 0:
        fig.delaxes(axs[batch_size // 2][1])


# ----------------------------------------------------------------------------------------------------
# Split dataset functions
# ----------------------------------------------------------------------------------------------------
def label_distribution(y: np.ndarray, labels=(0, 1, 2, 3)) -> np.ndarray:
    """
        Computes the label distribution for a given array of labels.

        Args:
            y (numpy.ndarray): Array of labels.
            labels (tuple, optional): Tuple of label values to compute the distribution for (default: (0, 1, 2, 3)).

        Returns:
            numpy.ndarray: Array of label distribution values.

        Created by ChatGPT.
    """
    return np.array([np.sum((y == label)) for label in labels]) / len(y)


def have_same_distributions(labels: np.ndarray, train_labels: np.ndarray, test_labels: np.ndarray, deviation: float = 0.1) -> bool:
    """
        Checks if the train and test label distributions are within the specified deviation of the complete label distribution.

        Args:
            labels (numpy.ndarray): Complete label array.
            train_labels (numpy.ndarray): Train label array.
            test_labels (numpy.ndarray): Test label array.
            deviation (float, optional): Maximum allowed deviation from the complete label distribution (default: 0.1).

        Returns:
            bool: True if train and test label distributions have similar distributions, False otherwise.

        Created by ChatGPT.
    """
    complete_distr = label_distribution(labels)
    test_distr = label_distribution(test_labels)
    train_distr = label_distribution(train_labels)

    upper_bound = (1 + deviation) * complete_distr
    lower_bound = (1 - deviation) * complete_distr

    d_test = (lower_bound < test_distr) & (test_distr < upper_bound)
    d_train = (lower_bound < train_distr) & (train_distr < upper_bound)

    return d_test.all() and d_train.all()


def split_dataset(fids: np.ndarray, labels: np.ndarray, data: list, seed: int = None) -> tuple[np.ndarray, np.ndarray, list, list]:
    """
        Splits a dataset into train and test sets based on the unique FIDs while maintaining label distribution.

        Args:
            fids (numpy.ndarray): Array of FIDs.
            labels (numpy.ndarray): Array of labels.
            data (list): List of data samples.
            seed (int, optional): Seed value for reproducible shuffling. Defaults to None.

        Returns:
            tuple: Tuple containing train indices, test indices, train data, and test data.

        Created by ChatGPT.
    """

    if seed is not None:
        np.random.seed(seed)

    # get unique fids
    unique_fids = np.unique(fids)
    length = len(labels)
    _20percent = length * 0.2
    _21percent = length * 0.21

    while True:
        # shuffle unique fids
        np.random.shuffle(unique_fids)
        test_indices = np.array([], dtype=int)

        # add all indices to the corresponding fid until the size is over 20%
        for fid in unique_fids:
            eq = np.where(fids == fid)[0]
            test_indices = np.append(test_indices, eq)

            if test_indices.shape[0] >= _20percent:
                break

        # if size is over 21% begin anew
        # else check if label distribution is ok
        if test_indices.shape[0] < _21percent:
            # get all indices that are not in 'test_indices'
            _range = np.arange(length)
            train_indices = _range[~np.isin(_range, test_indices)]

            # get the corresponding labels
            test_labels = labels[test_indices]
            train_labels = labels[train_indices]

            # if the distribution of the labels do not match begin anew
            # else break the loop
            if have_same_distributions(labels, train_labels, test_labels):
                break

    train_data = []
    test_data = []

    for idx in train_indices:
        train_data.append(data[idx])

    for idx in test_indices:
        test_data.append(data[idx])

    return train_indices, test_indices, train_data, test_data


# ----------------------------------------------------------------------------------------------------
# Dataset Generator
# ----------------------------------------------------------------------------------------------------
class BaseDataGenerator(tf.keras.utils.Sequence):
    """
        Base class for data generators in TensorFlow.

        Args:
            x: Input data.
            y: Target labels.
            to_fit (bool, optional): Whether to generate data for training (default: True).
            batch_size (int, optional): Batch size (default: 32).
            shuffle (bool, optional): Whether to shuffle the data at the end of each epoch (default: True).

        Attributes:
            x: Input data.
            y: Target data.
            size: Size of the dataset.
            to_fit: Whether to generate data for training.
            batch_size: Batch size.
            shuffle: Whether to shuffle the data at the end of each epoch.
            indexes: Indexes of the data samples.

        Methods:
            __len__(): Returns the number of batches per epoch.
            __getitem__(index): Generates one batch of data.
            _generate_X(indexes): Generates the input data for the given indexes.
            _generate_y(indexes): Generates the target labels for the given indexes.
            get(index): Returns the input data for the given index.
            on_epoch_end(): Callback function called at the end of each epoch.
            transform(x): Transforms the input data (abstract method, needs to be implemented by subclasses).

        Created by ChatGPT.
    """

    def __init__(self, x, y, to_fit=True, batch_size=32, shuffle=True):
        self.x = x
        self.y = y
        self.size = len(y)

        self.to_fit = to_fit
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.indexes = np.arange(self.size)
        self.on_epoch_end()

    def __len__(self) -> int:
        """
            Returns the number of batches per epoch.

            Returns:
                int: Number of batches per epoch.

            Created by ChatGPT.
        """
        return self.size // self.batch_size

    def __getitem__(self, index: int) -> Any:
        """
            Generates one batch of data.

            Args:
                index (int): Index of the batch.

            Returns:
                Any: Batch of data.

            Raises:
                IndexError: If the index is out of range.

            Created by ChatGPT.
        """
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        X = self._generate_X(indexes)

        if self.to_fit:
            y = self._generate_y(indexes)
            return X, y

        return X

    def _generate_X(self, indexes: Iterable) -> Any:
        """
            Generates the input data for the given indexes.

            Args:
                indexes (Iterable): Indexes of the data samples.

            Returns:
                Any: Input data.

            Created by ChatGPT.
        """
        X = [self.x[idx] for idx in indexes]
        X = self.transform(X)

        return tf.convert_to_tensor(X)

    def _generate_y(self, indexes: np.ndarray) -> np.ndarray:
        """
            Generates the target labels for the given indexes.

            Args:
                indexes (numpy.ndarray): Indexes of the data samples.

            Returns:
                numpy.ndarray: Target labels.

            Created by ChatGPT.
        """
        return self.y[indexes]

    def get(self, index) -> Any:
        """
            Returns the input data for the given index.

            Args:
                index: Index of the data sample.

            Returns:
                Any: Input data for the given index.

            Created by ChatGPT.
        """
        return self._generate_X([index])

    def on_epoch_end(self):
        """
            Callback function called at the end of each epoch.
            Shuffles the indexes if shuffle is set to True.

            Created by ChatGPT.
        """
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def transform(self, X: list) -> Iterable:
        """
            Transforms the input data.
            This is an abstract method that needs to be implemented by subclasses.

            Args:
                X (list): Input data.

            Returns:
                list: Transformed input data.

            Raises:
                NotImplementedError: If the method is not implemented by subclasses.

            Created by ChatGPT.
        """
        raise NotImplementedError('Subclass needs to implement transform method!')


# ----------------------------------------------------------------------------------------------------
# Custom metric and loss function
# ----------------------------------------------------------------------------------------------------
# Weight matrix
f, r = 2, 5
A = tf.constant([
    [0, r, r, 2 * r],
    [f, 0, f + r, r],
    [f, f + r, 0, r],
    [2 * f, f, f, 0]
], shape=(4, 4), dtype=tf.float32)


def weighted_loss(y_true, y_pred):
    """
        Computes the weighted loss between the true labels (y_true) and predicted labels (y_pred) using the weight matrix A.

        Args:
            y_true: Tensor, true labels.
            y_pred: Tensor, predicted labels.

        Returns:
            Tensor, mean weighted loss.

        Created by ChatGPT.
    """
    rows = tf.keras.backend.dot(y_true, A)
    batch_loss = tf.keras.backend.batch_dot(rows, y_pred)
    return tf.keras.backend.mean(batch_loss)


# For more information see: https://arxiv.org/abs/2008.05756
def matthews_correlation(C: np.ndarray) -> np.float64:
    """
        Computes the Matthews correlation coefficient (MCC) based on the confusion matrix C.

        Args:
            C: ndarray, confusion matrix.

        Returns:
            float64, Matthews correlation coefficient.

        Created by ChatGPT.
    """

    # Otherwise the values might overflow
    C = C.astype(np.int64)

    c = np.diag(C).sum()
    n = C.sum()
    p = C.sum(axis=0)
    t = C.sum(axis=1)

    return (c * n - p.dot(t)) / np.sqrt(((n ** 2 - p.dot(p)) * (n ** 2 - t.dot(t))))
