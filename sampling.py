import os, sys
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util import html
from data import CreateFtTLoader
from util import util
import torchvision.utils as tvutil
import torch
import torch.nn.functional as F
import numpy as np


if __name__ == '__main__':
    opt = TestOptions().parse()
    assert (opt.no_dropout)

    opt.results_dir = opt.checkpoints_dir
    test_data_loader = CreateFtTLoader(opt, is_test=True)

    model = create_model(opt)
    model.setup(opt)

    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    # sampling multiple times
    visuals, losses = model.validation(test_data_loader, how_many_to_valid=1024, n_samples=opt.n_samples)

    save_images(webpage, visuals, f'sampling ({opt.n_samples} samples)', aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)

    webpage.save()
