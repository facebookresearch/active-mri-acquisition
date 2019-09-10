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
            masks_dir=options.masks_dir)
        return train_loader, valid_loader
    else:
        test_loader = get_test_loader(
            batch_size=options.batchSize,
            num_workers=0,
            pin_memory=True,
            which_dataset=options.dataroot,
            mask_type=options.mask_type)
        return test_loader


def find_dataset_using_name(dataset_name):
    # Given the option --dataset [datasetname],
    # the file "datasets/datasetname_dataset.py"
    # will be imported.
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    # In the file, the class called DatasetNameDataset() will
    # be instantiated. It has to be a subclass of BaseDataset,
    # and it is case-insensitive.
    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        print(f"In {dataset_filename}.py, there should be a subclass of BaseDataset with class "
              f"name that matches {target_dataset_name} in lowercase.")
        exit(0)

    return dataset

def get_option_setter(dataset_name):
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options
