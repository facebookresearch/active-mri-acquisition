# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from . import base_data_loader


def create_data_loaders(options, is_test=False):

    if not is_test:
        train_loader, valid_loader = base_data_loader.get_train_valid_loader(
            options.dataset_dir,
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
        test_loader = base_data_loader.get_test_loader(
            options.dataset_dir,
            batch_size=options.batchSize,
            num_workers=0,
            pin_memory=True,
            which_dataset=options.dataroot,
            mask_type=options.mask_type,
            rnl_params=options.rnl_params,
        )
        return test_loader
