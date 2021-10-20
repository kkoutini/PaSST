import hashlib
import os
import time

import torch
from torch.utils.data import Dataset

from os.path import expanduser

import logging

def h6(w):
    return hashlib.md5(w.encode('utf-8')).hexdigest()[:6]

class AudioPreprocessDataset(Dataset):
    """A bases preprocessing dataset representing a Dataset of files that are loaded and preprossessed on the fly.

    Access elements via __getitem__ to return: preprocessor(x),sample_id,label

    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __init__(self, files, labels, label_encoder, base_dir, preprocessor, return_tensor=True, ordered_ids=None):
        self.files = files
        if ordered_ids is None:
            ordered_ids = files
        else:
            print("AudioPreprocessDataset: ordered_ids is not None using it instead of files !!!")
        self.ordered_ids = ordered_ids
        self.labels = labels
        self.label_encoder = label_encoder
        self.base_dir = base_dir
        self.preprocessor = preprocessor
        self.return_tensor = return_tensor

    def __getitem__(self, index):
        x = self.preprocessor(self.base_dir + self.files[index])
        if self.return_tensor and not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x)
        return x, self.ordered_ids[index], self.labels[index]

    def get_ordered_ids(self):
        return self.ordered_ids

    def get_ordered_labels(self):
        return self.labels

    def __len__(self):
        return len(self.ordered_ids)

class ObjectCacher:
    def __init__(self, get_obj_func, dataset_name, obj_name="",
                 cache_path="~/shared/kofta_cached_datasets/", verbose=True):
        self.dataset_name = dataset_name
        self.obj_name = obj_name
        cache_path = expanduser(cache_path)
        self.cache_path = os.path.join(cache_path, dataset_name)
        try:
            startTime = time.time()
            xpath = self.get_obj_cache_path()

            if verbose:
                logging.info(
                    "attempting to load x from cache at " + xpath + "...")
            self.obj = torch.load(xpath)

            if verbose:
                endTime = time.time()
                logging.info(
                    "loaded " + xpath + " from cache in %s " % (endTime - startTime))
        except IOError:
            if verbose:
                logging.info(
                    "Invalid cache " + xpath + " , recomputing")
            self.obj = get_obj_func()
            saveStartTime = time.time()
            dirpath=os.path.dirname(xpath)
            try:
                original_umask = os.umask(0)
                os.makedirs(dirpath, exist_ok=True)
            finally:
                os.umask(original_umask)
            torch.save(self.obj, xpath)
            if verbose:
                endTime = time.time()
                logging.info(
                    "loaded " + obj_name + " in %s, and cached in %s, total %s seconds " % (
                        (saveStartTime - startTime),
                        (endTime - saveStartTime), (endTime - startTime)))

    def get_obj_cache_path(self):
        return os.path.join(self.cache_path, self.obj_name + "_obj.pt")

    def get(self):
        return self.obj



class PreprocessDataset(Dataset):
    """A bases preprocessing dataset representing a preprocessing step of a Dataset preprossessed on the fly.


    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __init__(self, dataset, preprocessor):
        self.dataset = dataset
        if not callable(preprocessor):
            print("preprocessor: ", preprocessor)
            raise ValueError('preprocessor should be callable')
        self.preprocessor = preprocessor
    def __getitem__(self, index):
        return self.preprocessor(self.dataset[index])
    def __len__(self):
        return len(self.dataset)



class FilesCachedDataset(Dataset):
    def __init__(self, get_dataset_func, dataset_name, x_name="",
                 cache_path="~/shared/kofta_cached_datasets/",
                 ):
        """
            Cached the dataset in small torch.save files (1 file per sample).
            The dataset is suitable for SSDs being used bcache from a slow harddrive with a small
        @param get_dataset_func: fuction gets called if the file cache is invalid
        @param dataset_name: the folder containing the dataset
        @param x_name: tag for the version
        @param cache_path: cache_path
        """
        self.dataset = None

        def getDataset():
            if self.dataset == None:
                self.dataset = get_dataset_func()
            return self.dataset

        self.get_dataset_func = getDataset
        self.x_name = x_name
        cache_path = expanduser(cache_path)
        self.cache_path = os.path.join(cache_path, dataset_name, "files_cache", self.x_name)
        try:
            original_umask = os.umask(0)
            os.makedirs(self.cache_path, exist_ok=True)
        finally:
            os.umask(original_umask)

    def __getitem__(self, index):
        cpath = os.path.join(self.cache_path, str(index) + ".pt")
        try:
            return torch.load(cpath)
        except FileNotFoundError:
            tup = self.get_dataset_func()[index]
            torch.save(tup, cpath)
            return tup

    def get_ordered_labels(self):
        return self.get_dataset_func().get_ordered_labels()

    def get_ordered_ids(self):
        return self.get_dataset_func().get_ordered_ids()

    def get_xcache_path(self):
        return os.path.join(self.cache_path, self.x_name + "_x.pt")

    def get_ycache_path(self):
        return os.path.join(self.cache_path, self.y_name + "_y.pt")

    def get_sidcache_path(self):
        return os.path.join(self.cache_path, self.y_name + "_sid.pt")

    def __len__(self):
        return len(self.get_dataset_func())



class SelectionDataset(Dataset):
    """A dataset that selects a subsample from a dataset based on a set of sample ids.


        supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __init__(self, dataset, sample_ids):
        self.available_indexes = []
        self.dataset = dataset
        self.reselect(sample_ids)
        self.sample_ids = sample_ids

    def reselect(self, sample_ids):
        reverse_dict = dict([(sid, i) for i, sid in enumerate(self.dataset.get_ordered_ids())])
        self.available_indexes = [reverse_dict[sid] for sid in sample_ids]

    def get_ordered_ids(self):
        return self.sample_ids

    def get_ordered_labels(self):
        labels=self.dataset.get_ordered_labels()
        return [labels[i] for i in self.available_indexes]
        #raise NotImplementedError("Maybe reconsider caching only a selection Dataset. why not select after cache?")

    def __getitem__(self, index):
        return self.dataset[self.available_indexes[index]]

    def __len__(self):
        return len(self.available_indexes)

class SimpleSelectionDataset(Dataset):
    """A dataset that selects a subsample from a dataset based on a set of sample ids.


        supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __init__(self, dataset, available_indexes ):
        self.available_indexes = available_indexes
        self.dataset = dataset

    def __getitem__(self, index):
        return self.dataset[self.available_indexes[index]]

    def __len__(self):
        return len(self.available_indexes)

