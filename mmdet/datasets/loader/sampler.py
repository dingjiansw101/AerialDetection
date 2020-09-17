from __future__ import division

import math
import torch
import numpy as np
from collections import defaultdict

from torch.distributed import get_world_size, get_rank
from torch.utils.data import Sampler
from torch.utils.data import DistributedSampler as _DistributedSampler


class DistributedSampler(_DistributedSampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


class GroupSampler(Sampler):

    def __init__(self, dataset, samples_per_gpu=1):
        assert hasattr(dataset, 'flag')
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.flag = dataset.flag.astype(np.int64)
        self.group_sizes = np.bincount(self.flag)
        self.num_samples = 0
        for i, size in enumerate(self.group_sizes):
            self.num_samples += int(np.ceil(
                size / self.samples_per_gpu)) * self.samples_per_gpu

    def __iter__(self):
        indices = []
        for i, size in enumerate(self.group_sizes):
            if size == 0:
                continue
            indice = np.where(self.flag == i)[0]
            assert len(indice) == size
            np.random.shuffle(indice)
            num_extra = int(np.ceil(size / self.samples_per_gpu)
                            ) * self.samples_per_gpu - len(indice)
            indice = np.concatenate([indice, indice[:num_extra]])
            indices.append(indice)
        indices = np.concatenate(indices)
        indices = [
            indices[i * self.samples_per_gpu:(i + 1) * self.samples_per_gpu]
            for i in np.random.permutation(
                range(len(indices) // self.samples_per_gpu))
        ]
        indices = np.concatenate(indices)
        indices = torch.from_numpy(indices).long()
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples


class DistributedGroupSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self,
                 dataset,
                 samples_per_gpu=1,
                 num_replicas=None,
                 rank=None):
        if num_replicas is None:
            num_replicas = get_world_size()
        if rank is None:
            rank = get_rank()
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        assert hasattr(self.dataset, 'flag')
        self.flag = self.dataset.flag
        self.group_sizes = np.bincount(self.flag)

        self.num_samples = 0
        for i, j in enumerate(self.group_sizes):
            self.num_samples += int(
                math.ceil(self.group_sizes[i] * 1.0 / self.samples_per_gpu /
                          self.num_replicas)) * self.samples_per_gpu
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        for i, size in enumerate(self.group_sizes):
            if size > 0:
                indice = np.where(self.flag == i)[0]
                assert len(indice) == size
                indice = indice[list(torch.randperm(int(size),
                                                    generator=g))].tolist()
                extra = int(
                    math.ceil(
                        size * 1.0 / self.samples_per_gpu / self.num_replicas)
                ) * self.samples_per_gpu * self.num_replicas - len(indice)
                indice += indice[:extra]
                indices += indice

        assert len(indices) == self.total_size

        indices = [
            indices[j] for i in list(
                torch.randperm(
                    len(indices) // self.samples_per_gpu, generator=g))
            for j in range(i * self.samples_per_gpu, (i + 1) *
                           self.samples_per_gpu)
        ]

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset:offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

class DistributedRepeatFactorTrainingSampler(Sampler):
    """
    Similar to TrainingSampler, but a sample may appear more times than others based
    on its "repeat factor". This is suitable for training on class imbalanced datasets like LVIS.
    """

    def __init__(self,
                 dataset,
                 repeat_factors,
                 samples_per_gpu=1,
                 num_replicas=None,
                 rank=None):
        """
        Args:
            repeat_factors (Tensor): a float vector, the repeat factor for each indice. When it's
                full of ones, it is equivalent to ``TrainingSampler(len(repeat_factors), ...)``.
        """

        if num_replicas is None:
            num_replicas = get_world_size()
        if rank is None:
            rank = get_rank()
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        
        assert hasattr(self.dataset, 'flag')
        self.flag = self.dataset.flag
        self.group_sizes = np.bincount(self.flag)

        self.num_samples = 0
        for i, j in enumerate(self.group_sizes):
            self.num_samples += int(
                math.ceil(self.group_sizes[i] * 1.0 / self.samples_per_gpu /
                          self.num_replicas)) * self.samples_per_gpu
        self.total_size = self.num_samples * self.num_replicas
        ###

        self._world_size = comm.get_world_size()
        ##

        # Split into whole number (_int_part) and fractional (_frac_part) parts.
        self._int_part = torch.trunc(repeat_factors)
        self._frac_part = repeat_factors - self._int_part
        
    @staticmethod
    def repeat_factors_from_category_frequency(dataset_dicts, repeat_thresh=0.001):
        """
        Compute (fractional) per-image repeat factors based on category frequency.
        The repeat factor for an image is a function of the frequency of the rarest
        category labeled in that image. The "frequency of category c" in [0, 1] is defined
        as the fraction of images in the training set (without repeats) in which category c
        appears.
        See :paper:`lvis` (>= v2) Appendix B.2.
        Args:
            dataset_dicts (list[dict]): annotations in Detectron2 dataset format.
            repeat_thresh (float): frequency threshold below which data is repeated.
                If the frequency is half of `repeat_thresh`, the image will be
                repeated twice.
        Returns:
            torch.Tensor: the i-th element is the repeat factor for the dataset image
                at index i.
        """
        # 1. For each category c, compute the fraction of images that contain it: f(c)
        category_freq = defaultdict(int)
        for dataset_dict in dataset_dicts:  # For each image (without repeats)
            cat_ids = {ann["category_id"] for ann in dataset_dict["annotations"]}
            for cat_id in cat_ids:
                category_freq[cat_id] += 1
        num_images = len(dataset_dicts)
        for k, v in category_freq.items():
            category_freq[k] = v / num_images

        # 2. For each category c, compute the category-level repeat factor:
        #    r(c) = max(1, sqrt(t / f(c)))
        category_rep = {
            cat_id: max(1.0, math.sqrt(repeat_thresh / cat_freq))
            for cat_id, cat_freq in category_freq.items()
        }

        # 3. For each image I, compute the image-level repeat factor:
        #    r(I) = max_{c in I} r(c)
        rep_factors = []
        for dataset_dict in dataset_dicts:
            cat_ids = {ann["category_id"] for ann in dataset_dict["annotations"]}
            rep_factor = max({category_rep[cat_id] for cat_id in cat_ids})
            rep_factors.append(rep_factor)

        return torch.tensor(rep_factors, dtype=torch.float32)

    def _get_epoch_indices(self, generator):
        """
        Create a list of dataset indices (with repeats) to use for one epoch.
        Args:
            generator (torch.Generator): pseudo random number generator used for
                stochastic rounding.
        Returns:
            torch.Tensor: list of dataset indices to use in one epoch. Each index
                is repeated based on its calculated repeat factor.
        """
        # Since repeat factors are fractional, we use stochastic rounding so
        # that the target repeat factor is achieved in expectation over the
        # course of training
        rands = torch.rand(len(self._frac_part), generator=generator)
        rep_factors = self._int_part + (rands < self._frac_part).float()
        # Construct a list of indices in which we repeat images as specified
        indices = []
        for dataset_index, rep_factor in enumerate(rep_factors):
            indices.extend([dataset_index] * int(rep_factor.item()))
        return torch.tensor(indices, dtype=torch.int64)

    def __iter__(self):
        offset = self.rank
        yield from itertools.islice(self._infinite_indices(), offset, None, self.num_replicas)

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self.seed)
        while True:
            # Sample indices with repeats determined by stochastic rounding; each
            # "epoch" may have a slightly different size due to the rounding.
            indices = self._get_epoch_indices(g)
            randperm = torch.randperm(len(indices), generator=g)
            yield from indices[randperm]

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
