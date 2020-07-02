import importlib
from .base_data_loader import get_train_valid_loader, get_test_loader
from data.base_dataset import BaseDataset


def create_data_loaders(options, is_test=False):

    if not is_test:
        train_loader, valid_loader = get_train_valid_loader(
            batch_size=options.batchSize,
            num_workers=options.nThreads,
            pin_memory=True,
            which_dataset=options.dataroot,
            mask_type=options.mask_type,
            rnl_params=options.rnl_params,
            num_volumes_train=options.num_volumes_train,
            num_volumes_val=options.num_volumes_val,
        )
        return train_loader, valid_loader
    else:
        test_loader = get_test_loader(
            batch_size=options.batchSize,
            num_workers=0,
            pin_memory=True,
            which_dataset=options.dataroot,
            mask_type=options.mask_type,
            rnl_params=options.rnl_params,
        )
        return test_loader
