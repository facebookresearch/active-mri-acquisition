import importlib
from data.base_dataset import BaseDataset
from data.ft_data_loader import ft_data_loader


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
        print("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (
            dataset_filename, target_dataset_name))
        exit(0)

    return dataset


def get_option_setter(dataset_name):    
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_data_loaders(options, is_test=False):
    
    if not is_test:
        trainloader, validloader = ft_data_loader.get_train_valid_loader(
            batch_size=options.batchSize,
            num_workers=options.nThreads,
            pin_memory=True,
            which_dataset=options.dataroot
        )
        return trainloader, validloader
    else:
        testloader = ft_data_loader.get_test_loader(
            batch_size=options.batchSize,
            num_workers=0,
            pin_memory=True,
            which_dataset=options.dataroot
        )
        return testloader
