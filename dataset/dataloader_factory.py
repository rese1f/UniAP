import torch
from torch.utils.data import DataLoader
import os
from einops import rearrange, repeat

from .UniASET import UniASETHybridDataset, UniASETContinuousDataset
from .UniASET_constants import TASKS, TASKS_GROUP_DICT, TASKS_GROUP_TRAIN, TASKS_GROUP_NAMES
from .utils import crop_arrays


base_sizes = {
    224: (256, 256)
}


def get_train_loader(config, pin_memory=True, verbose=True, get_support_data=False):
    '''
    Load training dataloader.
    '''
    # set dataset size
    if get_support_data:
        dset_size = config.shot
    elif config.no_eval:
        dset_size = config.n_steps*config.global_batch_size
    else:
        dset_size = config.val_iter*config.global_batch_size

    # compute common arguments
    common_kwargs = {
        'base_size': base_sizes[config.img_size],
        'img_size': (config.img_size, config.img_size),
        'dset_size': dset_size,
        'seed': config.seed + int(os.environ.get('LOCAL_RANK', 0)),
        'precision': config.precision,
        'root_dir': config.root_dir,
    }

    # create dataset for episodic training
    if config.stage == 0:
        tasks = TASKS if config.task == 'all' else TASKS_GROUP_TRAIN[config.task_id]
        if verbose:
            print(f'Loading tasks {", ".join(tasks)} in train split.')

        # create training dataset.
        train_data = UniASETHybridDataset(
            tasks=tasks,
            shot=config.shot,
            tasks_per_batch=config.max_channels,
            domains_per_batch=config.domains_per_batch,
            image_augmentation=config.image_augmentation,
            unary_augmentation=config.unary_augmentation,
            binary_augmentation=config.binary_augmentation,
            mixed_augmentation=config.mixed_augmentation,
            **common_kwargs,
        )
     # create dataset for fine-tuning or testing
    else:
        if config.task in ['', 'all']:
            raise ValueError("task should be specified for fine-tuning")
        
        # create training dataset.
        train_data = UniASETHybridDataset(
            tasks=[config.task],
            shot=config.shot,
            tasks_per_batch=config.max_channels,
            domains_per_batch=config.domains_per_batch,
            image_augmentation=config.image_augmentation,
            unary_augmentation=config.unary_augmentation,
            binary_augmentation=config.binary_augmentation,
            mixed_augmentation=config.mixed_augmentation,
            **common_kwargs,
        )



    # create training loader.
    train_loader = DataLoader(train_data, batch_size=(config.global_batch_size // torch.cuda.device_count()),
                              shuffle=False, pin_memory=pin_memory,
                              drop_last=True, num_workers=config.num_workers)
        
    return train_loader


def get_eval_loader(config, task, split='valid', channel_idx=-1, pin_memory=True, verbose=True):
    '''
    Load evaluation dataloader.
    '''
    # no crop for evaluation.
    img_size = base_size = base_sizes[config.img_size]
        
    # evaluate some subset or the whole data.
    if config.n_eval_batches > 0:
        dset_size = config.n_eval_batches * config.eval_batch_size
    else:
        dset_size = -1
    
    # common arguments for both continuous and segmentation datasets.
    common_kwargs = {
        'root_dir': config.root_dir,
        'dset_size': dset_size,
        'base_size': base_size,
        'img_size': img_size,
        'seed': int(os.environ.get('LOCAL_RANK', 0)),
        'precision': config.precision,
        'shot': config.shot,
    }
    if verbose:
        if channel_idx < 0:
            print(f'Loading task {task} in {split} split.')
        else:
            print(f'Loading task {task}_{channel_idx} in {split} split.')
    
    # create appropriate dataset.
    eval_data = UniASETContinuousDataset(
        task=task,
        channel_idx=channel_idx,
        **common_kwargs
    )

    # create dataloader.
    eval_loader = DataLoader(eval_data, batch_size=(config.eval_batch_size // torch.cuda.device_count()),
                             shuffle=False, pin_memory=pin_memory,
                             drop_last=False, num_workers=1)
    
    return eval_loader


def get_validation_loaders(config, verbose=True):
    '''
    Load validation loaders (of unseen images) for training tasks.
    '''
    if config.stage == 0:
        if config.task == 'all':
            train_tasks = TASKS_GROUP_DICT
        else:
            train_tasks = TASKS_GROUP_TRAIN[config.task_id]
        loader_tag = 'mtrain_valid'
    else:
        if config.task in ['', 'all']:
            raise ValueError("task should be specified for fine-tuning")
        train_tasks = [config.task]
        loader_tag = 'mtest_valid'

    valid_loaders = {}
    for task in train_tasks:
        valid_loaders[task] = get_eval_loader(config, task, 'valid', verbose=verbose)
    
    return valid_loaders, loader_tag

  