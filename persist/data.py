import h5py
import pickle
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class ExpressionDataset(Dataset):
    '''
    Gene expression dataset capable of using subsets of inputs.

    Args:
      data: array of inputs with size (samples, dim).
      labels: array of labels with size (samples,) or (samples, dim).
    '''

    def __init__(self, data, labels):
        self.input_size = data.shape[1]
        self._data = data.astype(np.float32)
        if len(labels.shape) == 1:
            # Classification labels.
            self.output_size = len(np.unique(labels))
            self._output = labels.astype(np.long)
        else:
            # Regression labels.
            self.output_size = labels.shape[1]
            self._output = labels.astype(np.float32)
        self.set_inds(None)
        self.set_output_inds(None)

    def set_inds(self, inds, delete_remaining=False):
        '''
        Set input indices to be returned.

        Args:
          inds: list/array of selected indices.
          delete_remaining: whether to permanently delete the indices that are
            not selected.
        '''
        if inds is None:
            assert not delete_remaining
            inds = np.arange(self._data.shape[1])

            # Set input and inds.
            self.inds = inds
            self.data = self._data
            self.input_size = len(inds)
        else:
            # Verify inds.
            inds = np.sort(inds)
            assert len(inds) > 0
            assert (inds[0] >= 0) and (inds[-1] < self._data.shape[1])
            assert np.all(np.unique(inds, return_counts=True)[1] == 1)
            if len(inds) == self.input_size:
                assert not delete_remaining

            self.input_size = len(inds)
            if delete_remaining:
                # Reset data and input size.
                self._data = self._data[:, inds]
                self.data = self._data
                self.inds = np.arange(len(inds))
            else:
                # Set input and inds.
                self.inds = inds
                self.data = self._data[:, inds]

    def set_output_inds(self, inds, delete_remaining=False):
        '''
        Set output inds to be returned. Only for use with multivariate outputs.

        Args:
          inds: list/array of selected indices.
          delete_remaining: whether to permanently delete the indices that are
            not selected.
        '''
        if inds is None:
            assert not delete_remaining

            # Set output and inds.
            self.output = self._output
            if len(self._output.shape) == 1:
                # Classification labels.
                self.output_size = len(np.unique(self._output))
                self.output_inds = None
            else:
                # Regression labels.
                self.output_size = self._output.shape[1]
                self.output_inds = np.arange(self.output_size)
        else:
            # Verify that there are multiple output inds.
            assert len(self._output.shape) == 2

            # Verify inds.
            inds = np.sort(inds)
            assert len(inds) > 0
            assert (inds[0] >= 0) and (inds[-1] < self._output.shape[1])
            assert np.all(np.unique(inds, return_counts=True)[1] == 1)
            if len(inds) == self.output_size:
                assert not delete_remaining

            self.output_size = len(inds)
            if delete_remaining:
                # Reset data and input size.
                self._output = self._output[:, inds]
                self.output = self._output
                self.output_inds = np.arange(len(inds))
            else:
                # Set output and inds.
                self.output_inds = inds
                self.output = self._output[:, inds]

    @property
    def max_input_size(self):
        return self._data.shape[1]

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        return self.data[index], self.output[index]


class HDF5ExpressionDataset(Dataset):
    '''
    Dataset wrapper, capable of using subsets of inputs.

    Args:
      filename: HDF5 filename.
      data_name: key for data array.
      label_name: key for labels.
      sample_inds: list of indices for rows to be sampled.
      initialize: whether to initialize by opening the HDF5 file. This should
        only be done when using a data loader with no worker threads.
    '''

    def __init__(self, filename, data_name, label_name,
                 sample_inds=None, initialize=False):
        # Set up data variables.
        self.filename = filename
        self.data_name = data_name
        self.label_name = label_name

        # Set sample inds.
        hf = h5py.File(filename, 'r')
        data = hf[self.data_name]
        labels = hf[self.label_name]
        if sample_inds is None:
            sample_inds = np.arange(len(data))
        self.sample_inds = sample_inds

        # Set input, output size.
        self.input_size = data.shape[1]
        if labels.ndim == 1:
            # Classification labels.
            self.output_size = len(np.unique(labels))
            self.multiple_outputs = False
        else:
            # Regression labels.
            self.output_size = labels.shape[1]
            self.multiple_outputs = True
        hf.close()

        # Set input inds.
        self.all_inds = np.arange(self.input_size)
        self.set_inds(None)

        # Set output inds.
        if self.multiple_outputs:
            self.all_output_inds = np.arange(self.output_size)
        else:
            self.all_output_inds = None
        self.set_output_inds(None)

        # Initialize.
        if initialize:
            self.init_worker(0)

    def set_inds(self, inds, delete_remaining=False):
        '''
        Set input indices to be returned.

        Args:
          inds: list/array of selected indices.
          delete_remaining: whether to permanently delete the indices that are
            not selected.
        '''
        if inds is None:
            assert not delete_remaining
            self.inds = np.arange(len(self.all_inds))
            self.relative_inds = self.all_inds
            self.input_size = len(self.all_inds)
        else:
            # Verify inds.
            inds = np.sort(inds)
            assert len(inds) > 0
            assert (inds[0] >= 0) and (inds[-1] < len(self.all_inds))
            assert np.all(np.unique(inds, return_counts=True)[1] == 1)

            self.input_size = len(inds)
            if delete_remaining:
                self.inds = np.arange(len(inds))
                self.all_inds = self.all_inds[inds]
                self.relative_inds = self.all_inds
            else:
                self.inds = inds
                self.relative_inds = self.all_inds[inds]

    def set_output_inds(self, inds, delete_remaining=False):
        '''
        Set output inds to be returned. Only for use with multivariate outputs.

        Args:
          inds: list/array of selected indices.
          delete_remaining: whether to permanently delete the indices that are
            not selected.
        '''
        if inds is None:
            assert not delete_remaining
            if self.multiple_outputs:
                self.output_inds = np.arange(len(self.all_output_inds))
                self.output_relative_inds = self.all_output_inds
                self.output_size = len(self.all_output_inds)
            else:
                self.output_inds = None
                self.output_relative_inds = None
        else:
            # Verify that there are multiple output inds.
            assert self.multiple_outputs

            # Verify inds.
            inds = np.sort(inds)
            assert len(inds) > 0
            assert (inds[0] >= 0) and (inds[-1] < len(self.all_output_inds))
            assert np.all(np.unique(inds, return_counts=True)[1] == 1)
            if len(inds) == self.output_size:
                assert not delete_remaining

            self.output_size = len(inds)
            if delete_remaining:
                self.output_inds = np.arange(len(inds))
                self.all_output_inds = self.all_output_inds[inds]
                self.output_relative_inds = self.all_output_inds
            else:
                self.output_inds = inds
                self.output_relative_inds = self.all_output_inds[inds]

    def init_worker(self, worker_id):
        '''Initialize worker in data loader thread.'''
        self.h5 = h5py.File(self.filename, 'r', swmr=True)

    @property
    def max_input_size(self):
        return len(self.all_inds)

    def __len__(self):
        return len(self.sample_inds)

    def __getitem__(self, index):
        # Possibly initialize worker to open HDF5 file.
        if not hasattr(self, 'h5'):
            self.init_worker(0)

        index = self.sample_inds[index]
        data = self.h5[self.data_name][index][self.relative_inds]
        labels = self.h5[self.label_name][index]
        if self.output_relative_inds is not None:
            labels = labels[self.output_relative_inds]
        return data, labels


def split_data(data, seed=123, val_portion=0.1, test_portion=0.1):
    '''Split data into train, val, test.'''
    N = data.shape[0]
    N_val = int(val_portion * N)
    N_test = int(test_portion * N)
    train, test = train_test_split(data, test_size=N_test, random_state=seed)
    train, val = train_test_split(train, test_size=N_val, random_state=seed+1)
    return train, val, test


def bootstrapped_dataset(dataset, seed=None):
    '''Sample a bootstrapped dataset.'''
    if isinstance(dataset, ExpressionDataset):
        data = dataset.data
        labels = dataset.output
        if seed:
            np.random.seed(seed)
        N = len(data)
        inds = np.random.choice(N, size=N, replace=True)
        return ExpressionDataset(data[inds], labels[inds])
    elif isinstance(dataset, HDF5ExpressionDataset):
        inds = dataset.sample_inds
        inds = np.random.choice(inds, size=len(inds), replace=True)
        return HDF5ExpressionDataset(dataset.filename, dataset.data_name,
                                     dataset.label_name, sample_inds=inds)
    else:
        raise ValueError('dataset must be ExpressionDataset or '
                         'HDF5ExpressionDataset')


class GeneSet:
    '''
    Set of genes, represented by their indices and names.

    Args:
      inds: gene indices.
      names: gene names.
    '''

    def __init__(self, inds, names):
        self.inds = inds
        self.names = names

    def save(self, filename):
        '''Save object by pickling.'''
        with open(filename, 'wb') as f:
            pickle.dump(self, f)


def load_set(filename):
    '''Load GeneSet object.'''
    with open(filename, 'rb') as f:
        subset = pickle.load(f)
    if isinstance(subset, GeneSet):
        return subset
    else:
        raise ValueError('object is not GeneSet')
