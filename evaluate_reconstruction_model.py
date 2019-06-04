from options.test_options import TestOptions
from data import CreateFtTLoader
from models import create_model

if __name__ == '__main__':
    opts = TestOptions().parse()

    train_data_loader, val_data_loader = CreateFtTLoader(opts, valid_size=0.9)

    dataset_size = len(train_data_loader)
    # Load model
    model = create_model(opts)
    model.setup(opts)
    model.eval()

    model.isTrain = True    # Hack because model code bypasses random masks if this is False
    model.opt.dynamic_mask_type = 'random_lowfreq'  # 'random_lowfreq'  # 'random'
    visuals, losses = model.validation(val_data_loader, set_validation_phase=False)
