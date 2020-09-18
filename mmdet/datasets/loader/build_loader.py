from functools import partial

from mmcv.runner import get_dist_info
from mmcv.parallel import collate
from torch.utils.data import DataLoader

from .sampler import GroupSampler, DistributedGroupSampler, DistributedSampler
from .sampler import DistributedRepeatFactorTrainingSampler

# https://github.com/pytorch/pytorch/issues/973
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


def build_dataloader(dataset,
                     imgs_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     repeat_samples=False,
                     **kwargs):
    shuffle = kwargs.get('shuffle', True)
    if dist:
        rank, world_size = get_dist_info()
        if shuffle:
            if repeat_samples:
                repeat_factors = DistributedRepeatFactorTrainingSampler.repeat_factors_from_category_frequency(kwargs['dataset_dicts'])
                sampler = DistributedRepeatFactorTrainingSampler(dataset, repeat_factors, imgs_per_gpu,
                                              world_size, rank)

            else:
                sampler = DistributedGroupSampler(dataset, imgs_per_gpu,
                                                world_size, rank)
        else:
            sampler = DistributedSampler(
                dataset, world_size, rank, shuffle=False)
        batch_size = imgs_per_gpu
        num_workers = workers_per_gpu
    else:
        sampler = GroupSampler(dataset, imgs_per_gpu) if shuffle else None
        batch_size = num_gpus * imgs_per_gpu
        num_workers = num_gpus * workers_per_gpu

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=imgs_per_gpu),
        pin_memory=False)

    return data_loader
